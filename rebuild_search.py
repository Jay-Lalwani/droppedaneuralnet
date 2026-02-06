#!/usr/bin/env python3
"""
Permutation search for dropped residual network.

Pieces:
- 48 Up:   48 -> 96  (weight [96,48])
- 48 Down: 96 -> 48  (weight [48,96])
- 1 Last:  48 -> 1   (weight [1,48])

Pipeline per restart:
Stage 1: Pair Up/Down into blocks (Hungarian if SciPy else greedy).
Stage 2: Build diverse candidate orders (randomized greedy + 2opt), pick best by MSE on subsample.
Stage 3: Alternating refinement:
    - ORDER-SA: SA over block order (true MSE)
    - PAIR-SA:  SA over pairing (swap/cycle/shuffle downs), order fixed (true MSE)

Global: run --restarts times, keep best full MSE.
"""

import os, re, csv, time, math, random, argparse, logging
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F

# -----------------------------
# Logging / Seeding
# -----------------------------
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def now_ms():
    return int(time.time() * 1000)

# -----------------------------
# Load pieces + data
# -----------------------------
Piece = Dict[str, torch.Tensor]

def load_pieces(pieces_dir: str, device: torch.device):
    pieces: Dict[int, Piece] = {}
    up_idxs, down_idxs = [], []
    last_idx = None

    files = sorted(os.listdir(pieces_dir))
    for f in files:
        if not f.endswith(".pth"):
            continue
        m = re.search(r"piece_(\d+)\.pth", f)
        if not m:
            continue
        idx = int(m.group(1))
        sd = torch.load(os.path.join(pieces_dir, f), map_location="cpu")
        w = sd["weight"].to(device=device, dtype=torch.float32)
        b = sd["bias"].to(device=device, dtype=torch.float32)

        pieces[idx] = {"weight": w, "bias": b}

        if tuple(w.shape) == (96, 48):
            up_idxs.append(idx)
        elif tuple(w.shape) == (48, 96):
            down_idxs.append(idx)
        elif tuple(w.shape) == (1, 48):
            last_idx = idx
        else:
            logging.warning(f"Unknown piece shape for idx={idx}: {tuple(w.shape)}")

    if last_idx is None:
        raise ValueError("Could not find last layer (expected weight shape [1,48]).")
    if len(up_idxs) != len(down_idxs):
        raise ValueError(f"Up/Down count mismatch: up={len(up_idxs)} down={len(down_idxs)}")

    up_idxs.sort()
    down_idxs.sort()
    return pieces, up_idxs, down_idxs, last_idx

def load_data_csv(data_file: str, device: torch.device, target_col: str = "pred"):
    xs, ys = [], []
    with open(data_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            feats = [float(row[f"measurement_{i}"]) for i in range(48)]
            y = float(row[target_col])
            xs.append(feats)
            ys.append(y)
    x = torch.tensor(xs, dtype=torch.float32, device=device)
    y = torch.tensor(ys, dtype=torch.float32, device=device).unsqueeze(1)
    return x, y

# -----------------------------
# Forward / Loss
# -----------------------------
def forward_from_piece_perm(x: torch.Tensor, pieces: Dict[int, Piece], piece_perm: List[int]) -> torch.Tensor:
    curr = x
    for k in range(0, len(piece_perm) - 1, 2):
        up_idx = piece_perm[k]
        down_idx = piece_perm[k + 1]
        w_up, b_up = pieces[up_idx]["weight"], pieces[up_idx]["bias"]
        w_dn, b_dn = pieces[down_idx]["weight"], pieces[down_idx]["bias"]

        residual = curr
        hidden = F.linear(curr, w_up, b_up)
        hidden = F.relu(hidden)
        out = F.linear(hidden, w_dn, b_dn)
        curr = residual + out

    last_idx = piece_perm[-1]
    w_last, b_last = pieces[last_idx]["weight"], pieces[last_idx]["bias"]
    return F.linear(curr, w_last, b_last)

@torch.no_grad()
def mse_for_perm(pieces: Dict[int, Piece], piece_perm: List[int], x: torch.Tensor, y: torch.Tensor) -> float:
    y_pred = forward_from_piece_perm(x, pieces, piece_perm)
    return float(torch.mean((y_pred - y) ** 2).item())

def blocks_to_piece_perm(block_order: List[int], block_pairs: List[Tuple[int, int]], last_idx: int) -> List[int]:
    perm: List[int] = []
    for bi in block_order:
        u, d = block_pairs[bi]
        perm.extend([u, d])
    perm.append(last_idx)
    return perm

@torch.no_grad()
def mse_for_block_order(
    pieces: Dict[int, Piece],
    block_pairs: List[Tuple[int, int]],
    block_order: List[int],
    last_idx: int,
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    piece_perm = blocks_to_piece_perm(block_order, block_pairs, last_idx)
    return mse_for_perm(pieces, piece_perm, x, y)

# -----------------------------
# Stage 1: Pairing via Assignment
# -----------------------------
@torch.no_grad()
def pairing_cost_matrix(
    x_batch: torch.Tensor,
    pieces: Dict[int, Piece],
    up_idxs: List[int],
    down_idxs: List[int],
    act_weight: float = 0.05,
    update_weight: float = 1.0,
    mean_weight: float = 0.05,
) -> torch.Tensor:
    n_up = len(up_idxs)
    n_dn = len(down_idxs)

    W_dn = torch.stack([pieces[j]["weight"] for j in down_idxs], dim=0)  # [n_dn,48,96]
    b_dn = torch.stack([pieces[j]["bias"]   for j in down_idxs], dim=0)  # [n_dn,48]

    C = torch.empty((n_up, n_dn), device=x_batch.device, dtype=torch.float32)

    for i, ui in enumerate(up_idxs):
        w_up, b_up = pieces[ui]["weight"], pieces[ui]["bias"]
        pre = F.linear(x_batch, w_up, b_up)                      # [B,96]
        act = (pre > 0).float().mean().item()
        act_pen = abs(act - 0.5)

        h = F.relu(pre)
        out_all = torch.einsum("bm,nkm->nbk", h, W_dn) + b_dn[:, None, :]  # [n_dn,B,48]

        update_mse = out_all.pow(2).mean(dim=(1, 2))
        mean_pen   = out_all.mean(dim=(1, 2)).pow(2)

        C[i] = update_weight * update_mse + act_weight * act_pen + mean_weight * mean_pen

    return C

def solve_assignment(cost: torch.Tensor) -> List[int]:
    cost_cpu = cost.detach().cpu().numpy()
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(cost_cpu)
        logging.info("Pairing: Hungarian (scipy) linear_sum_assignment")
        match = [-1] * cost_cpu.shape[0]
        for ri, ci in zip(r, c):
            match[int(ri)] = int(ci)
        return match
    except Exception as e:
        logging.warning(f"Pairing: Hungarian unavailable/failed ({e}); using greedy fallback")
        n = cost_cpu.shape[0]
        used = set()
        match = [-1] * n
        for i in range(n):
            best_j, best_v = None, float("inf")
            for j in range(n):
                if j in used:
                    continue
                v = float(cost_cpu[i, j])
                if v < best_v:
                    best_v, best_j = v, j
            used.add(best_j)
            match[i] = best_j
        return match

def stage1_pair_blocks(
    pieces: Dict[int, Piece],
    up_idxs: List[int],
    down_idxs: List[int],
    x_for_cost: torch.Tensor,
) -> List[Tuple[int, int]]:
    logging.info("Stage 1: pairing Up/Down into Blocks (assignment)")
    C = pairing_cost_matrix(x_for_cost, pieces, up_idxs, down_idxs)
    match = solve_assignment(C)
    return [(up_idxs[i], down_idxs[j]) for i, j in enumerate(match)]

# -----------------------------
# Stage 2: Ordering via Transition Costs (diverse candidates)
# -----------------------------
@torch.no_grad()
def block_health_costs(
    x: torch.Tensor,
    W_up: torch.Tensor, b_up: torch.Tensor,
    W_dn: torch.Tensor, b_dn: torch.Tensor,
    act_weight: float = 0.05,
    update_weight: float = 1.0,
    mean_weight: float = 0.05,
) -> torch.Tensor:
    pre = torch.einsum("nd,bod->bno", x, W_up) + b_up[:, None, :]         # [B,N,96]
    act = (pre > 0).float().mean(dim=(1, 2))
    act_pen = (act - 0.5).abs()

    h = F.relu(pre)
    out = torch.einsum("bni,bki->bnk", h, W_dn) + b_dn[:, None, :]

    update_mse = out.pow(2).mean(dim=(1, 2))
    mean_pen   = out.mean(dim=(1, 2)).pow(2)
    return update_weight * update_mse + act_weight * act_pen + mean_weight * mean_pen

@torch.no_grad()
def apply_single_block(
    x: torch.Tensor,
    w_up: torch.Tensor, b_up: torch.Tensor,
    w_dn: torch.Tensor, b_dn: torch.Tensor,
) -> torch.Tensor:
    pre = F.linear(x, w_up, b_up)
    h = F.relu(pre)
    out = F.linear(h, w_dn, b_dn)
    return x + out

@torch.no_grad()
def transition_cost_matrix(
    x_seed: torch.Tensor,
    pieces: Dict[int, Piece],
    block_pairs: List[Tuple[int, int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    B = len(block_pairs)
    device = x_seed.device

    W_up = torch.stack([pieces[u]["weight"] for (u, _) in block_pairs], dim=0)
    b_up = torch.stack([pieces[u]["bias"]   for (u, _) in block_pairs], dim=0)
    W_dn = torch.stack([pieces[d]["weight"] for (_, d) in block_pairs], dim=0)
    b_dn = torch.stack([pieces[d]["bias"]   for (_, d) in block_pairs], dim=0)

    start_cost = block_health_costs(x_seed, W_up, b_up, W_dn, b_dn)

    w = torch.empty((B, B), device=device, dtype=torch.float32)
    for i in range(B):
        u_i, d_i = block_pairs[i]
        x_i = apply_single_block(
            x_seed,
            pieces[u_i]["weight"], pieces[u_i]["bias"],
            pieces[d_i]["weight"], pieces[d_i]["bias"],
        )
        w[i] = block_health_costs(x_i, W_up, b_up, W_dn, b_dn)

    return start_cost, w

def two_opt_improve(order: List[int], start_cost: torch.Tensor, w: torch.Tensor, max_passes: int = 2) -> List[int]:
    def path_cost(ordr: List[int]) -> float:
        c = float(start_cost[ordr[0]].item())
        for i in range(len(ordr) - 1):
            c += float(w[ordr[i], ordr[i + 1]].item())
        return c

    best = order[:]
    best_c = path_cost(best)
    B = len(best)

    for _ in range(max_passes):
        improved = False
        for i in range(1, B - 2):
            for j in range(i + 1, B - 1):
                cand = best[:i] + list(reversed(best[i:j + 1])) + best[j + 1:]
                c = path_cost(cand)
                if c + 1e-12 < best_c:
                    best, best_c = cand, c
                    improved = True
        if not improved:
            break
    return best

def randomized_greedy_path(
    start_cost: torch.Tensor,
    w: torch.Tensor,
    rng: random.Random,
    topk_start: int = 6,
    topk_next: int = 6,
    tau: float = 0.2,
) -> List[int]:
    """
    Diverse greedy builder:
      - start sampled among topk_start lowest start_cost
      - each step samples next among topk_next lowest transition costs from prev
    tau controls softness (lower -> more greedy).
    """
    B = int(start_cost.numel())
    # start choice
    sc = [(float(start_cost[i].item()), i) for i in range(B)]
    sc.sort()
    start_pool = [i for _, i in sc[:max(1, min(topk_start, B))]]

    # sample start uniformly (or could do softmax)
    start = start_pool[rng.randrange(len(start_pool))]

    order = [start]
    unused = set(range(B))
    unused.remove(start)

    while unused:
        prev = order[-1]
        candidates = [(float(w[prev, j].item()), j) for j in unused]
        candidates.sort()
        pool = candidates[:max(1, min(topk_next, len(candidates)))]

        # softmax sampling over pool using tau
        vals = [v for v, _ in pool]
        m = min(vals)
        probs = [math.exp(-(v - m) / max(tau, 1e-6)) for v in vals]
        s = sum(probs)
        r = rng.random() * s
        acc = 0.0
        chosen = pool[-1][1]
        for p, (_, j) in zip(probs, pool):
            acc += p
            if acc >= r:
                chosen = j
                break

        order.append(chosen)
        unused.remove(chosen)

    return order

def stage2_order_blocks_diverse(
    pieces: Dict[int, Piece],
    block_pairs: List[Tuple[int, int]],
    x_seed: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    last_idx: int,
    rng: random.Random,
    candidates: int = 12,
) -> List[int]:
    """
    Build several diverse orders using randomized greedy, locally improve with 2-opt,
    then pick the best by true MSE on (x_eval, y_eval).
    """
    logging.info(f"Stage 2: initial ordering (diverse candidates={candidates})")
    start_cost, w = transition_cost_matrix(x_seed, pieces, block_pairs)

    best_order = None
    best_mse = float("inf")

    for c in range(candidates):
        if c == 0:
            # baseline deterministic-ish: tau small and topk=1 makes it greedy
            ord0 = randomized_greedy_path(start_cost, w, rng, topk_start=1, topk_next=1, tau=0.01)
        else:
            ord0 = randomized_greedy_path(start_cost, w, rng, topk_start=6, topk_next=6, tau=0.2)

        ord0 = two_opt_improve(ord0, start_cost, w, max_passes=2)
        mse = mse_for_block_order(pieces, block_pairs, ord0, last_idx, x_eval, y_eval)

        if mse < best_mse:
            best_mse = mse
            best_order = ord0

    assert best_order is not None
    logging.info(f"Stage 2: best candidate sub-MSE = {best_mse:.12f}")
    return best_order

# -----------------------------
# Stage 3a: ORDER-SA (stronger move set)
# -----------------------------
def propose_order_move(order: List[int], rng: random.Random) -> List[int]:
    B = len(order)
    cand = order[:]
    r = rng.random()

    if r < 0.35:
        # swap
        i, j = rng.randrange(B), rng.randrange(B)
        cand[i], cand[j] = cand[j], cand[i]
    elif r < 0.65:
        # reverse segment
        i, j = sorted([rng.randrange(B), rng.randrange(B)])
        if i != j:
            cand[i:j + 1] = reversed(cand[i:j + 1])
    elif r < 0.85:
        # relocate one element
        i = rng.randrange(B)
        v = cand.pop(i)
        j = rng.randrange(B)
        cand.insert(j, v)
    else:
        # shuffle a short window (bigger jump)
        win = rng.randrange(4, 10)  # 4..9
        i = rng.randrange(0, max(1, B - win))
        window = cand[i:i + win]
        rng.shuffle(window)
        cand[i:i + win] = window

    return cand

def order_sa(
    pieces: Dict[int, Piece],
    block_pairs: List[Tuple[int, int]],
    last_idx: int,
    init_order: List[int],
    x_full: torch.Tensor, y_full: torch.Tensor,
    x_sub: torch.Tensor,  y_sub: torch.Tensor,
    iters: int,
    T0: float,
    Tmin: float,
    eval_full_every: int,
    rng: random.Random,
) -> Tuple[List[int], float]:
    def temperature(t: int) -> float:
        frac = t / max(1, iters - 1)
        return T0 * ((Tmin / T0) ** frac)

    curr = init_order[:]
    curr_loss = mse_for_block_order(pieces, block_pairs, curr, last_idx, x_sub, y_sub)

    best = curr[:]
    best_sub = curr_loss
    best_full = mse_for_block_order(pieces, block_pairs, best, last_idx, x_full, y_full)

    logging.info(f"    init: sub_mse={best_sub:.10f} full_mse={best_full:.10f}")

    last_log = now_ms()
    for t in range(1, iters + 1):
        T = temperature(t)
        cand = propose_order_move(curr, rng)
        cand_loss = mse_for_block_order(pieces, block_pairs, cand, last_idx, x_sub, y_sub)

        d = cand_loss - curr_loss
        accept = (d <= 0.0) or (rng.random() < math.exp(-d / max(T, 1e-12)))
        if accept:
            curr, curr_loss = cand, cand_loss
            if curr_loss < best_sub - 1e-15:
                best, best_sub = curr[:], curr_loss

        if t % eval_full_every == 0:
            full = mse_for_block_order(pieces, block_pairs, best, last_idx, x_full, y_full)
            if full < best_full - 1e-15:
                best_full = full

        if t % 250 == 0:
            now = now_ms()
            dt = (now - last_log) / 1000.0
            last_log = now
            logging.info(
                f"    t={t:5d}/{iters} T={T:.2e} curr_sub={curr_loss:.10f} "
                f"best_sub={best_sub:.10f} best_full~={best_full:.10f} ({dt:.2f}s)"
            )
            if best_full <= 1e-15:
                logging.info("    Early stop: full MSE ~ 0.")
                break

    return best, best_full

# -----------------------------
# Stage 3b: PAIR-SA (stronger pairing moves)
# -----------------------------
def propose_pair_move(pairs: List[Tuple[int, int]], rng: random.Random) -> List[Tuple[int, int]]:
    B = len(pairs)
    cand = pairs[:]
    r = rng.random()

    if r < 0.55:
        # swap downs between 2 blocks
        i, j = rng.randrange(B), rng.randrange(B)
        while j == i:
            j = rng.randrange(B)
        ui, di = cand[i]
        uj, dj = cand[j]
        cand[i] = (ui, dj)
        cand[j] = (uj, di)

    elif r < 0.80:
        # 3-cycle of downs among 3 blocks
        i, j, k = rng.sample(range(B), 3)
        ui, di = cand[i]
        uj, dj = cand[j]
        uk, dk = cand[k]
        cand[i] = (ui, dj)
        cand[j] = (uj, dk)
        cand[k] = (uk, di)

    else:
        # shuffle downs within a subset of blocks
        ksz = rng.randrange(4, 8)  # 4..7
        idxs = rng.sample(range(B), ksz)
        downs = [cand[i][1] for i in idxs]
        rng.shuffle(downs)
        for ii, new_d in zip(idxs, downs):
            cand[ii] = (cand[ii][0], new_d)

    return cand

def pair_sa(
    pieces: Dict[int, Piece],
    init_pairs: List[Tuple[int, int]],
    fixed_order: List[int],
    last_idx: int,
    x_full: torch.Tensor, y_full: torch.Tensor,
    x_sub: torch.Tensor,  y_sub: torch.Tensor,
    iters: int,
    T0: float,
    Tmin: float,
    eval_full_every: int,
    rng: random.Random,
) -> Tuple[List[Tuple[int, int]], float]:
    def temperature(t: int) -> float:
        frac = t / max(1, iters - 1)
        return T0 * ((Tmin / T0) ** frac)

    @torch.no_grad()
    def loss_sub(pairs):
        perm = blocks_to_piece_perm(fixed_order, pairs, last_idx)
        return mse_for_perm(pieces, perm, x_sub, y_sub)

    @torch.no_grad()
    def loss_full(pairs):
        perm = blocks_to_piece_perm(fixed_order, pairs, last_idx)
        return mse_for_perm(pieces, perm, x_full, y_full)

    curr = init_pairs[:]
    curr_loss = loss_sub(curr)

    best = curr[:]
    best_sub = curr_loss
    best_full = loss_full(best)

    logging.info(f"    init: sub_mse={best_sub:.10f} full_mse={best_full:.10f}")

    last_log = now_ms()
    for t in range(1, iters + 1):
        T = temperature(t)
        cand = propose_pair_move(curr, rng)
        cand_loss = loss_sub(cand)

        d = cand_loss - curr_loss
        accept = (d <= 0.0) or (rng.random() < math.exp(-d / max(T, 1e-12)))
        if accept:
            curr, curr_loss = cand, cand_loss
            if curr_loss < best_sub - 1e-15:
                best, best_sub = curr[:], curr_loss

        if t % eval_full_every == 0:
            full = loss_full(best)
            if full < best_full - 1e-15:
                best_full = full

        if t % 250 == 0:
            now = now_ms()
            dt = (now - last_log) / 1000.0
            last_log = now
            logging.info(
                f"    t={t:5d}/{iters} T={T:.2e} curr_sub={curr_loss:.10f} "
                f"best_sub={best_sub:.10f} best_full~={best_full:.10f} ({dt:.2f}s)"
            )
            if best_full <= 1e-15:
                logging.info("    Early stop: full MSE ~ 0.")
                break

    return best, best_full

# -----------------------------
# Batching helpers (per restart)
# -----------------------------
def make_batches(
    x_full: torch.Tensor,
    y_full: torch.Tensor,
    pair_batch: int,
    order_batch: int,
    sub_batch: int,
    seed: int,
    device: torch.device,
):
    n = x_full.shape[0]
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    rp = torch.randperm(n, generator=gen, device=device)
    x_pair = x_full[rp[:min(pair_batch, n)]]
    x_order = x_full[rp[:min(order_batch, n)]]

    rp2 = torch.randperm(n, generator=gen, device=device)
    x_sub = x_full[rp2[:min(sub_batch, n)]]
    y_sub = y_full[rp2[:min(sub_batch, n)]]

    return x_pair, x_order, x_sub, y_sub

# -----------------------------
# Main
# -----------------------------
def main():
    setup_logger()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces_dir", type=str, default="pieces")
    ap.add_argument("--data_file", type=str, default="historical_data.csv")
    ap.add_argument("--target_col", type=str, default="pred", choices=["pred", "true"])
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])

    ap.add_argument("--seed", type=int, default=0, help="Use -1 for random seed each run")
    ap.add_argument("--restarts", type=int, default=1)

    ap.add_argument("--pair_batch", type=int, default=1024)
    ap.add_argument("--order_batch", type=int, default=1024)
    ap.add_argument("--sub_batch", type=int, default=5000)

    ap.add_argument("--stage2_candidates", type=int, default=16)

    ap.add_argument("--alt_rounds", type=int, default=6)

    ap.add_argument("--order_sa_iters", type=int, default=8000)
    ap.add_argument("--order_sa_T0", type=float, default=3e-2)
    ap.add_argument("--order_sa_Tmin", type=float, default=1e-5)

    ap.add_argument("--pair_sa_iters", type=int, default=8000)
    ap.add_argument("--pair_sa_T0", type=float, default=2e-2)
    ap.add_argument("--pair_sa_Tmin", type=float, default=2e-5)

    ap.add_argument("--eval_full_every", type=int, default=100)

    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    else:
        device = torch.device(args.device)

    logging.info(f"Using device: {device}")

    pieces, up_idxs, down_idxs, last_idx = load_pieces(args.pieces_dir, device)
    logging.info(f"Loaded {len(pieces)} pieces: up={len(up_idxs)} down={len(down_idxs)} last={last_idx}")

    x_full, y_full = load_data_csv(args.data_file, device, target_col=args.target_col)
    logging.info(f"Loaded data: x={tuple(x_full.shape)} y={tuple(y_full.shape)} target_col={args.target_col}")

    global_best_mse = float("inf")
    global_best_perm: Optional[List[int]] = None

    base_seed = args.seed
    if base_seed == -1:
        base_seed = int(time.time()) ^ (os.getpid() << 8)

    for r in range(args.restarts):
        run_seed = base_seed + 100003 * r
        seed_all(run_seed)
        rng = random.Random(run_seed + 424242)

        logging.info("=" * 72)
        logging.info(f"RESTART {r+1}/{args.restarts} | seed={run_seed}")

        x_pair, x_order, x_sub, y_sub = make_batches(
            x_full, y_full,
            args.pair_batch, args.order_batch, args.sub_batch,
            seed=run_seed + 7,
            device=device,
        )

        # Stage 1
        t0 = time.time()
        block_pairs = stage1_pair_blocks(pieces, up_idxs, down_idxs, x_pair)
        naive_order = list(range(len(block_pairs)))
        mse_stage1 = mse_for_block_order(pieces, block_pairs, naive_order, last_idx, x_full, y_full)
        logging.info(f"Stage 1 done in {time.time()-t0:.2f}s | full MSE (naive order) = {mse_stage1:.12f}")

        # Stage 2 (diverse)
        t0 = time.time()
        order = stage2_order_blocks_diverse(
            pieces=pieces,
            block_pairs=block_pairs,
            x_seed=x_order,
            x_eval=x_sub,
            y_eval=y_sub,
            last_idx=last_idx,
            rng=rng,
            candidates=args.stage2_candidates,
        )
        mse_stage2 = mse_for_block_order(pieces, block_pairs, order, last_idx, x_full, y_full)
        logging.info(f"Stage 2 done in {time.time()-t0:.2f}s | full MSE = {mse_stage2:.12f}")

        best_mse = mse_stage2
        best_perm = blocks_to_piece_perm(order, block_pairs, last_idx)
        pairs = block_pairs

        # Stage 3 (alternating)
        for rr in range(1, args.alt_rounds + 1):
            logging.info(f"Alt round {rr}/{args.alt_rounds}: ORDER-SA")
            t0 = time.time()
            order, full_after_order = order_sa(
                pieces=pieces,
                block_pairs=pairs,
                last_idx=last_idx,
                init_order=order,
                x_full=x_full, y_full=y_full,
                x_sub=x_sub,  y_sub=y_sub,
                iters=args.order_sa_iters,
                T0=args.order_sa_T0,
                Tmin=args.order_sa_Tmin,
                eval_full_every=args.eval_full_every,
                rng=rng,
            )
            logging.info(f"  After ORDER-SA: full MSE = {full_after_order:.12f} (dt={time.time()-t0:.2f}s)")
            if full_after_order < best_mse:
                best_mse = full_after_order
                best_perm = blocks_to_piece_perm(order, pairs, last_idx)
                logging.info(f"  NEW BEST (restart-local) after ORDER-SA: {best_mse:.12f}")

            logging.info(f"Alt round {rr}/{args.alt_rounds}: PAIR-SA (swap/cycle/shuffle downs)")
            t0 = time.time()
            pairs, full_after_pair = pair_sa(
                pieces=pieces,
                init_pairs=pairs,
                fixed_order=order,
                last_idx=last_idx,
                x_full=x_full, y_full=y_full,
                x_sub=x_sub,  y_sub=y_sub,
                iters=args.pair_sa_iters,
                T0=args.pair_sa_T0,
                Tmin=args.pair_sa_Tmin,
                eval_full_every=args.eval_full_every,
                rng=rng,
            )
            logging.info(f"  After PAIR-SA:  full MSE = {full_after_pair:.12f} (dt={time.time()-t0:.2f}s)")
            if full_after_pair < best_mse:
                best_mse = full_after_pair
                best_perm = blocks_to_piece_perm(order, pairs, last_idx)
                logging.info(f"  NEW BEST (restart-local) after PAIR-SA: {best_mse:.12f}")

            if best_mse <= 1e-15:
                logging.info("Reached ~0 MSE; stopping early.")
                break

        logging.info(f"RESTART {r+1} best full MSE = {best_mse:.12f}")

        if best_mse < global_best_mse:
            global_best_mse = best_mse
            global_best_perm = best_perm
            logging.info(f"NEW GLOBAL BEST: full MSE = {global_best_mse:.12f}")
            logging.info(f"NEW GLOBAL BEST PERMUTATION: {global_best_perm}")

        if global_best_mse <= 1e-15:
            break

    logging.info("=" * 72)
    logging.info(f"FINAL GLOBAL BEST full MSE = {global_best_mse:.12f}")
    print("\nBEST PIECE PERMUTATION (length={}):".format(len(global_best_perm)))
    print(global_best_perm)
    print("\nBest full MSE:", global_best_mse)

if __name__ == "__main__":
    main()
