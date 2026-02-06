#!/usr/bin/env python3
"""
Rebuild a "dropped" network whose pieces are individual nn.Linear layers:
- 48 Up layers: 48 -> 96   (weight shape [96,48])
- 48 Down layers: 96 -> 48 (weight shape [48,96])
- 1 Last layer:  48 -> 1   (weight shape [1,48])

Pipeline (with logging):
Stage 1: Pair Up/Down into residual Blocks via assignment (Hungarian if SciPy available).
Stage 2: Initial ordering via transition-cost surrogate (greedy + 2-opt).
Stage 3: Alternating refinement:
    - ORDER-SA on true MSE (permute block order)
    - PAIR-SA on true MSE (swap Down partners between blocks, order fixed)
    Repeat a few rounds.

Outputs best piece permutation (length 97): [up0,down0, up1,down1, ..., up47,down47, last]
"""

import os, re, csv, time, math, random, argparse, logging
from typing import Dict, List, Tuple

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
def forward_from_piece_perm(
    x: torch.Tensor,
    pieces: Dict[int, Piece],
    piece_perm: List[int],
) -> torch.Tensor:
    """
    piece_perm: length 97 = [up0,down0, up1,down1, ..., up47,down47, last]
    """
    curr = x
    for k in range(0, len(piece_perm) - 1, 2):
        up_idx = piece_perm[k]
        down_idx = piece_perm[k + 1]
        w_up, b_up = pieces[up_idx]["weight"], pieces[up_idx]["bias"]
        w_dn, b_dn = pieces[down_idx]["weight"], pieces[down_idx]["bias"]

        residual = curr
        hidden = F.linear(curr, w_up, b_up)  # [B,96]
        hidden = F.relu(hidden)
        out = F.linear(hidden, w_dn, b_dn)   # [B,48]
        curr = residual + out

    last_idx = piece_perm[-1]
    w_last, b_last = pieces[last_idx]["weight"], pieces[last_idx]["bias"]
    y_pred = F.linear(curr, w_last, b_last)  # [B,1]
    return y_pred

@torch.no_grad()
def mse_for_perm(
    pieces: Dict[int, Piece],
    piece_perm: List[int],
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    y_pred = forward_from_piece_perm(x, pieces, piece_perm)
    return float(torch.mean((y_pred - y) ** 2).item())

def blocks_to_piece_perm(
    block_order: List[int],
    block_pairs: List[Tuple[int, int]],
    last_idx: int
) -> List[int]:
    perm: List[int] = []
    for bi in block_order:
        u, d = block_pairs[bi]
        perm.extend([u, d])
    perm.append(last_idx)
    return perm

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
    """
    Cost matrix C[i,j] lower is better pairing (up_i with down_j).
    Combines:
      - update magnitude E||down(ReLU(up(x)))||^2
      - activation rate penalty |P(pre>0)-0.5|
      - mean output penalty (bias blowups)
    """
    n_up = len(up_idxs)
    n_dn = len(down_idxs)

    W_dn = torch.stack([pieces[j]["weight"] for j in down_idxs], dim=0)  # [n_dn,48,96]
    b_dn = torch.stack([pieces[j]["bias"]   for j in down_idxs], dim=0)  # [n_dn,48]

    C = torch.empty((n_up, n_dn), device=x_batch.device, dtype=torch.float32)

    for i, ui in enumerate(up_idxs):
        w_up, b_up = pieces[ui]["weight"], pieces[ui]["bias"]             # [96,48], [96]
        pre = F.linear(x_batch, w_up, b_up)                               # [B,96]
        act = (pre > 0).float().mean().item()
        act_pen = abs(act - 0.5)

        h = F.relu(pre)                                                   # [B,96]
        out_all = torch.einsum("bm,nkm->nbk", h, W_dn) + b_dn[:, None, :]  # [n_dn,B,48]

        update_mse = out_all.pow(2).mean(dim=(1, 2))                       # [n_dn]
        mean_pen   = out_all.mean(dim=(1, 2)).pow(2)                       # [n_dn]

        C[i] = update_weight * update_mse + act_weight * act_pen + mean_weight * mean_pen

    return C

def solve_assignment(cost: torch.Tensor) -> List[int]:
    """
    Returns match[i] = j for each row i.
    Uses SciPy Hungarian if available; otherwise greedy fallback.
    """
    cost_cpu = cost.detach().cpu().numpy()
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(cost_cpu)
    match = [-1] * cost_cpu.shape[0]
    for ri, ci in zip(r, c):
        match[int(ri)] = int(ci)
    assert all(m >= 0 for m in match)
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
    block_pairs = [(up_idxs[i], down_idxs[j]) for i, j in enumerate(match)]
    return block_pairs

# -----------------------------
# Stage 2: Ordering via Transition Costs (greedy + 2opt)
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
    """
    Vectorized 'health' cost for applying each block to x.
    W_up: [B,96,48], b_up:[B,96], W_dn:[B,48,96], b_dn:[B,48]
    Returns costs [B]
    """
    pre = torch.einsum("nd,bod->bno", x, W_up) + b_up[:, None, :]         # [B,N,96]
    act = (pre > 0).float().mean(dim=(1, 2))                               # [B]
    act_pen = (act - 0.5).abs()

    h = F.relu(pre)                                                        # [B,N,96]
    out = torch.einsum("bni,bki->bnk", h, W_dn) + b_dn[:, None, :]         # [B,N,48]

    update_mse = out.pow(2).mean(dim=(1, 2))                                # [B]
    mean_pen   = out.mean(dim=(1, 2)).pow(2)                                # [B]
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
    """
    start_cost[j]: cost of using block j first on x_seed
    w[i,j]: cost of using block j after applying block i to x_seed
    """
    B = len(block_pairs)
    device = x_seed.device

    W_up = torch.stack([pieces[u]["weight"] for (u, _) in block_pairs], dim=0)  # [B,96,48]
    b_up = torch.stack([pieces[u]["bias"]   for (u, _) in block_pairs], dim=0)  # [B,96]
    W_dn = torch.stack([pieces[d]["weight"] for (_, d) in block_pairs], dim=0)  # [B,48,96]
    b_dn = torch.stack([pieces[d]["bias"]   for (_, d) in block_pairs], dim=0)  # [B,48]

    start_cost = block_health_costs(x_seed, W_up, b_up, W_dn, b_dn)            # [B]

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

def greedy_path(start_cost: torch.Tensor, w: torch.Tensor) -> List[int]:
    B = int(start_cost.numel())
    start = int(torch.argmin(start_cost).item())
    order = [start]
    unused = set(range(B))
    unused.remove(start)
    while unused:
        prev = order[-1]
        nxt = min(unused, key=lambda j: float(w[prev, j].item()))
        order.append(nxt)
        unused.remove(nxt)
    return order

def path_cost(order: List[int], start_cost: torch.Tensor, w: torch.Tensor) -> float:
    c = float(start_cost[order[0]].item())
    for i in range(len(order) - 1):
        c += float(w[order[i], order[i + 1]].item())
    return c

def two_opt_improve(order: List[int], start_cost: torch.Tensor, w: torch.Tensor, max_passes: int = 2) -> List[int]:
    best = order[:]
    best_c = path_cost(best, start_cost, w)
    B = len(best)

    for _ in range(max_passes):
        improved = False
        for i in range(1, B - 2):
            for j in range(i + 1, B - 1):
                cand = best[:i] + list(reversed(best[i:j + 1])) + best[j + 1:]
                c = path_cost(cand, start_cost, w)
                if c + 1e-12 < best_c:
                    best, best_c = cand, c
                    improved = True
        if not improved:
            break
    return best

def stage2_order_blocks(
    pieces: Dict[int, Piece],
    block_pairs: List[Tuple[int, int]],
    x_seed: torch.Tensor,
) -> List[int]:
    logging.info("Stage 2: initial ordering via transition costs (greedy + 2-opt)")
    start_cost, w = transition_cost_matrix(x_seed, pieces, block_pairs)
    order = greedy_path(start_cost, w)
    order = two_opt_improve(order, start_cost, w, max_passes=2)
    return order

# -----------------------------
# Stage 3a: ORDER-SA on true MSE
# -----------------------------
def propose_move(order: List[int], rng: random.Random) -> List[int]:
    B = len(order)
    cand = order[:]
    move = rng.random()
    if move < 0.40:
        i, j = rng.randrange(B), rng.randrange(B)
        cand[i], cand[j] = cand[j], cand[i]
    elif move < 0.70:
        i, j = sorted([rng.randrange(B), rng.randrange(B)])
        if i != j:
            cand[i:j + 1] = reversed(cand[i:j + 1])
    else:
        i = rng.randrange(B)
        v = cand.pop(i)
        j = rng.randrange(B)
        cand.insert(j, v)
    return cand

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

def stage3_refine_order_sa(
    pieces: Dict[int, Piece],
    block_pairs: List[Tuple[int, int]],
    last_idx: int,
    init_order: List[int],
    x_full: torch.Tensor, y_full: torch.Tensor,
    x_sub: torch.Tensor,  y_sub: torch.Tensor,
    iters: int = 6000,
    T0: float = 2e-2,
    Tmin: float = 1e-5,
    eval_full_every: int = 200,
    seed: int = 0,
) -> List[int]:
    logging.info("  ORDER-SA: simulated annealing on block order (subsampled + periodic full eval)")
    rng = random.Random(seed)

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
        cand = propose_move(curr, rng)
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

        if t % 200 == 0:
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

    return best

# -----------------------------
# Stage 3b: PAIR-SA (swap Down partners) on true MSE
# -----------------------------
def sa_pairing_swaps(
    pieces: Dict[int, Piece],
    block_pairs: List[Tuple[int, int]],   # (up, down)
    last_idx: int,
    fixed_order: List[int],
    x_full: torch.Tensor, y_full: torch.Tensor,
    x_sub: torch.Tensor,  y_sub: torch.Tensor,
    iters: int = 6000,
    T0: float = 5e-3,
    Tmin: float = 5e-6,
    eval_full_every: int = 250,
    seed: int = 0,
) -> List[Tuple[int, int]]:
    logging.info("  PAIR-SA: simulated annealing on pairing (swap Down partners), order fixed")
    rng = random.Random(seed)

    def temperature(t: int) -> float:
        frac = t / max(1, iters - 1)
        return T0 * ((Tmin / T0) ** frac)

    @torch.no_grad()
    def loss_on_sub(pairs):
        perm = blocks_to_piece_perm(fixed_order, pairs, last_idx)
        return mse_for_perm(pieces, perm, x_sub, y_sub)

    @torch.no_grad()
    def loss_on_full(pairs):
        perm = blocks_to_piece_perm(fixed_order, pairs, last_idx)
        return mse_for_perm(pieces, perm, x_full, y_full)

    curr = block_pairs[:]
    curr_loss = loss_on_sub(curr)

    best = curr[:]
    best_sub = curr_loss
    best_full = loss_on_full(best)

    logging.info(f"    init: sub_mse={best_sub:.10f} full_mse={best_full:.10f}")

    B = len(curr)
    last_log = now_ms()
    for t in range(1, iters + 1):
        T = temperature(t)
        i, j = rng.randrange(B), rng.randrange(B)
        while j == i:
            j = rng.randrange(B)

        cand = curr[:]
        ui, di = cand[i]
        uj, dj = cand[j]
        cand[i] = (ui, dj)
        cand[j] = (uj, di)

        cand_loss = loss_on_sub(cand)
        d = cand_loss - curr_loss
        accept = (d <= 0.0) or (rng.random() < math.exp(-d / max(T, 1e-12)))

        if accept:
            curr, curr_loss = cand, cand_loss
            if curr_loss < best_sub - 1e-15:
                best, best_sub = curr[:], curr_loss

        if t % eval_full_every == 0:
            full = loss_on_full(best)
            if full < best_full - 1e-15:
                best_full = full

        if t % 200 == 0:
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

    return best

# -----------------------------
# Orchestration
# -----------------------------
def main():
    setup_logger()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces_dir", type=str, default="pieces")
    ap.add_argument("--data_file", type=str, default="historical_data.csv")
    ap.add_argument("--target_col", type=str, default="pred", choices=["pred", "true"])
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--pair_batch", type=int, default=1024)
    ap.add_argument("--order_batch", type=int, default=1024)
    ap.add_argument("--sub_batch", type=int, default=5000)     # bigger by default to reduce subsample noise

    ap.add_argument("--order_sa_iters", type=int, default=6000)
    ap.add_argument("--order_sa_T0", type=float, default=2e-2)
    ap.add_argument("--order_sa_Tmin", type=float, default=1e-5)

    ap.add_argument("--pair_sa_iters", type=int, default=6000)
    ap.add_argument("--pair_sa_T0", type=float, default=5e-3)
    ap.add_argument("--pair_sa_Tmin", type=float, default=5e-6)

    ap.add_argument("--eval_full_every", type=int, default=100)  # more frequent full eval
    ap.add_argument("--alt_rounds", type=int, default=5)

    args = ap.parse_args()

    seed_all(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    else:
        device = torch.device(args.device)

    logging.info(f"Using device: {device}")

    pieces, up_idxs, down_idxs, last_idx = load_pieces(args.pieces_dir, device)
    logging.info(f"Loaded {len(pieces)} pieces: up={len(up_idxs)} down={len(down_idxs)} last={last_idx}")

    x_full, y_full = load_data_csv(args.data_file, device, target_col=args.target_col)
    logging.info(f"Loaded data: x={tuple(x_full.shape)} y={tuple(y_full.shape)} target_col={args.target_col}")

    n = x_full.shape[0]
    rp = torch.randperm(n, device=device)
    x_pair  = x_full[rp[:min(args.pair_batch, n)]]
    x_order = x_full[rp[:min(args.order_batch, n)]]

    rp2 = torch.randperm(n, device=device)
    x_sub = x_full[rp2[:min(args.sub_batch, n)]]
    y_sub = y_full[rp2[:min(args.sub_batch, n)]]

    # ----------------- Stage 1 -----------------
    t0 = time.time()
    block_pairs = stage1_pair_blocks(pieces, up_idxs, down_idxs, x_pair)
    init_order = list(range(len(block_pairs)))
    init_perm = blocks_to_piece_perm(init_order, block_pairs, last_idx)
    mse1 = mse_for_perm(pieces, init_perm, x_full, y_full)
    logging.info(f"Stage 1 done in {time.time()-t0:.2f}s | full MSE (paired, naive order) = {mse1:.12f}")

    # ----------------- Stage 2 -----------------
    t0 = time.time()
    order = stage2_order_blocks(pieces, block_pairs, x_order)
    perm2 = blocks_to_piece_perm(order, block_pairs, last_idx)
    mse2 = mse_for_perm(pieces, perm2, x_full, y_full)
    logging.info(f"Stage 2 done in {time.time()-t0:.2f}s | full MSE (surrogate order) = {mse2:.12f}")

    # ----------------- Stage 3 (Alternating) -----------------
    pairs = block_pairs
    best_mse = mse2
    best_perm = perm2

    for r in range(1, args.alt_rounds + 1):
        logging.info(f"Stage 3 Alt Round {r}/{args.alt_rounds}: ORDER-SA")
        t_round = time.time()
        order = stage3_refine_order_sa(
            pieces=pieces,
            block_pairs=pairs,
            last_idx=last_idx,
            init_order=order,
            x_full=x_full, y_full=y_full,
            x_sub=x_sub, y_sub=y_sub,
            iters=args.order_sa_iters,
            T0=args.order_sa_T0,
            Tmin=args.order_sa_Tmin,
            eval_full_every=args.eval_full_every,
            seed=args.seed + 1000 * r,
        )
        perm_tmp = blocks_to_piece_perm(order, pairs, last_idx)
        mse_tmp = mse_for_perm(pieces, perm_tmp, x_full, y_full)
        logging.info(f"  After ORDER-SA: full MSE = {mse_tmp:.12f} (round time {time.time()-t_round:.2f}s)")
        if mse_tmp < best_mse:
            best_mse, best_perm = mse_tmp, perm_tmp
            logging.info(f"  NEW BEST after ORDER-SA: full MSE = {best_mse:.12f}")

        logging.info(f"Stage 3 Alt Round {r}/{args.alt_rounds}: PAIR-SA (swap down partners)")
        t_round = time.time()
        pairs = sa_pairing_swaps(
            pieces=pieces,
            block_pairs=pairs,
            last_idx=last_idx,
            fixed_order=order,
            x_full=x_full, y_full=y_full,
            x_sub=x_sub, y_sub=y_sub,
            iters=args.pair_sa_iters,
            T0=args.pair_sa_T0,
            Tmin=args.pair_sa_Tmin,
            eval_full_every=args.eval_full_every,
            seed=args.seed + 1000 * r + 777,
        )
        perm_tmp = blocks_to_piece_perm(order, pairs, last_idx)
        mse_tmp = mse_for_perm(pieces, perm_tmp, x_full, y_full)
        logging.info(f"  After PAIR-SA:  full MSE = {mse_tmp:.12f} (round time {time.time()-t_round:.2f}s)")
        if mse_tmp < best_mse:
            best_mse, best_perm = mse_tmp, perm_tmp
            logging.info(f"  NEW BEST after PAIR-SA: full MSE = {best_mse:.12f}")

        if best_mse <= 1e-15:
            logging.info("Reached ~0 full MSE; stopping alternating refinement.")
            break

    logging.info(f"Final best full MSE = {best_mse:.12f}")

    print("\nBEST PIECE PERMUTATION (length={}):".format(len(best_perm)))
    print(best_perm)
    print("\nBest full MSE:", best_mse)

if __name__ == "__main__":
    main()
