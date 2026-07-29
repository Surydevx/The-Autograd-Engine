import os
import csv
import json

from AUTOGRAD.Engine import Value

path_name = "Model"

# =========================================================================================
# Loss Functions
# =========================================================================================

def bce_loss(predictions, targets):
    """Binary Cross-Entropy Loss for Classification with Sigmoid."""
    total_loss = Value(0.0)
    for p, y in zip(predictions, targets):
        p_sig = p.sigmoid()
        epsilon = Value(1e-5)
        term1 = Value(y) * (p_sig + epsilon).log()
        term2 = Value(1.0 - y) * (Value(1.0) - p_sig + epsilon).log()
        
        total_loss = total_loss + (term1 + term2)
        
    return -total_loss / Value(len(targets))


def mse_loss(predictions, targets):
    """Mean Squared Error Loss for Regression."""
    total_loss = Value(0.0)
    for p, y in zip(predictions, targets):
        error = p - Value(y)
        total_loss = total_loss + (error * error)
        
    return total_loss / Value(len(targets))

def categorical_cross_entropy(logits_list, target_indices):
    """
    Categorical Cross-Entropy for Multi-Class Classification.
    - logits_list: A list of lists, where each sub-list is the raw output of the MLP for one sample.
    - target_indices: A list of integers representing the correct class index.
    """
    total_loss = Value(0.0)
    
    for logits, target_idx in zip(logits_list, target_indices):
        max_logit = max([l.data for l in logits])
        exp_logits = [(l - Value(max_logit)).exp() for l in logits]
        sum_exp = sum(exp_logits, Value(0.0))
        probabilities = [e / sum_exp for e in exp_logits]
        correct_prob = probabilities[int(target_idx)]# implementtaion of softmax function.
        
        # Added epsilon to prevent log(0)
        epsilon = Value(1e-5)
        total_loss = total_loss + (correct_prob + epsilon).log() * Value(-1.0)
        
    return total_loss / Value(len(target_indices))


def hinge_loss(predictions, targets):
    """
    Hinge Loss for Support Vector Machine style classification.
    the targets must be -1 or 1, NOT 0 or 1!
    """
    total_loss = Value(0.0)
    
    for p, y in zip(predictions, targets):
        margin = Value(1.0) - (p * Value(y)) # Loss = max(0, 1 - y * p)
        total_loss = total_loss + margin.relu()
        
    return total_loss / Value(len(targets))


def l1_loss(predictions, targets):
    """
    Mean Absolute Error (L1 Loss) for Regression.
    Highly robust against extreme outliers compared to MSE.
    """
    total_loss = Value(0.0)
    
    for p, y in zip(predictions, targets):
        error = p - Value(y)
        
        if error.data > 0:
            abs_error = error
        else:
            abs_error = -error
            
        total_loss = total_loss + abs_error
        
    return total_loss / Value(len(targets))

# =========================================================================================
# Data Handling
# =========================================================================================

def load_csv(filepath):
    """Loads the dataset from a CSV file assuming the last column is the target label."""
    xs, ys = [], []
    path = os.path.expanduser(filepath)
    try:
        with open(path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader) # Skip header row
            for row in reader:
                if not row: 
                    continue
                xs.append([float(v) for v in row[:-1]]) # features
                ys.append(float(row[-1]))              # target
                
        print(f"Loaded {len(ys)} samples from {path}")
        return xs, ys
        
    except FileNotFoundError:
        print(f"Error: '{path}' not found. Falling back to default dataset.")
        return [], []

# =========================================================================================
# Telemetry
# =========================================================================================

def export_telemetry(loss_node, filename):
    """Captures nodes, edges, values, and gradients for Manim rendering iteratively."""
    export_path = os.path.expanduser(path_name)
    os.makedirs(export_path, exist_ok=True)    
    export_path = os.path.join(export_path, filename)
    
    nodes, edges = [], []
    visited = set()
    stack = [loss_node]

    while stack:
        v = stack.pop()
        v_id = id(v)
        
        if v_id not in visited:
            visited.add(v_id)
            
            nodes.append({
                "id": str(v_id),
                "label": getattr(v, "_operation", "input"), 
                "data": round(v.data, 4) if hasattr(v, 'data') else 0.0,
                "grad": round(v.gradient, 4) if hasattr(v, 'gradient') else 0.0
            })
            
            for child in getattr(v, '_previous', []):
                edges.append({"from": str(id(child)), "to": str(v_id)})
                stack.append(child)

    with open(export_path, "w") as f:
        json.dump({"nodes": nodes, "edges": edges}, f, indent=2)
        
    print(f"Telemetry saved to {filename}")