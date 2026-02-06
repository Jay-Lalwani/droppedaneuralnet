
import torch
import torch.nn.functional as F
import os
import re
import csv

# --- Configuration ---
pieces_dir = 'pieces'
data_file = 'historical_data.csv'

SOLUTION_PERMUTATION = [84, 70, 58, 67, 18, 8, 59, 25, 95, 63, 16, 66, 4, 11, 81, 22, 91, 29, 56, 21, 48, 7, 15, 30, 60, 46, 3, 57, 39, 96, 77, 89, 94, 19, 31, 47, 41, 36, 45, 20, 1, 24, 69, 51, 49, 40, 28, 26, 68, 12, 5, 72, 87, 90, 50, 76, 44, 32, 86, 6, 73, 54, 23, 52, 2, 38, 14, 53, 27, 75, 88, 33, 10, 83, 13, 71, 65, 78, 74, 92, 35, 17, 61, 82, 37, 80, 0, 34, 43, 93, 64, 55, 42, 9, 62, 79, 85]

def analyze_and_verify():
    print("--- 1. Identify Block Types ---")
    pieces = {}
    files = sorted(os.listdir(pieces_dir))
    
    for f in files:
        if not f.endswith('.pth'): continue
        idx = int(re.search(r'piece_(\d+).pth', f).group(1))
        path = os.path.join(pieces_dir, f)
        sd = torch.load(path, map_location='cpu')
        
        w = sd['weight']
        shape = w.shape
        
        # Classify based on shape
        if shape == torch.Size([96, 48]):
            p_type = "Block_Input (48->96)"
        elif shape == torch.Size([48, 96]):
            p_type = "Block_Output (96->48)"
        elif shape == torch.Size([1, 48]):
            p_type = "Last_Layer (48->1)"
        else:
            p_type = f"Unknown {shape}"
            
        pieces[idx] = {'type': p_type, 'params': sd}
        # Uncomment to print every piece
        # print(f"Piece {idx}: {p_type}")

    print(f"Loaded {len(pieces)} pieces.")
    
    # Verify the solution permutation matches types
    print("\n--- 2. Verifying Permutation Consistency ---")
    valid_structure = True
    
    # We expect In, Out, In, Out ... Last
    for i in range(0, len(SOLUTION_PERMUTATION)-1, 2):
        in_idx = SOLUTION_PERMUTATION[i]
        out_idx = SOLUTION_PERMUTATION[i+1]
        
        type_in = pieces[in_idx]['type']
        type_out = pieces[out_idx]['type']
        
        if "Block_Input" not in type_in or "Block_Output" not in type_out:
            print(f"Mismatch at index {i}: {type_in} -> {type_out}")
            valid_structure = False
            break
            
    last_idx = SOLUTION_PERMUTATION[-1]
    if "Last_Layer" not in pieces[last_idx]['type']:
        print(f"Last layer mismatch: {pieces[last_idx]['type']}")
        valid_structure = False
        
    if valid_structure:
        print("Structure Valid: (Input -> Output) * 48 -> LastLayer")
    else:
        print("Structure Invalid!")
        return

    print("\n--- 3. Measuring Difference (Loss) ---")
    # Load Data
    measurements = []
    targets = []
    with open(data_file, 'r') as f:
        reader = csv.reader(f)
        next(reader) # Skip header
        for row in reader:
            vals = [float(x) for x in row]
            measurements.append(vals[:48])
            targets.append(vals[48])
            
    x = torch.tensor(measurements)
    y_true = torch.tensor(targets).unsqueeze(1)
    
    # Run Model
    curr_x = x
    
    # Apply Blocks
    for i in range(0, len(SOLUTION_PERMUTATION)-1, 2):
        in_idx = SOLUTION_PERMUTATION[i]
        out_idx = SOLUTION_PERMUTATION[i+1]
        
        w_in = pieces[in_idx]['params']['weight']
        b_in = pieces[in_idx]['params']['bias']
        w_out = pieces[out_idx]['params']['weight']
        b_out = pieces[out_idx]['params']['bias']
        
        # Residual Block: x + linear_out(relu(linear_in(x)))
        residual = curr_x
        hidden = F.linear(curr_x, w_in, b_in)
        hidden = F.relu(hidden)
        out = F.linear(hidden, w_out, b_out)
        curr_x = residual + out
        
    # Apply Last Layer
    l_idx = SOLUTION_PERMUTATION[-1]
    w_last = pieces[l_idx]['params']['weight']
    b_last = pieces[l_idx]['params']['bias']
    
    y_pred = F.linear(curr_x, w_last, b_last)
    
    # Calculate MSE
    mse = torch.mean((y_pred - y_true)**2).item()
    print(f"Solution MSE: {mse:.6f}")

if __name__ == "__main__":
    analyze_and_verify()
