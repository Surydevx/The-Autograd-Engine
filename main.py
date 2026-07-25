import argparse
import matplotlib.pyplot as plt
from rich.live import Live
from rich.table import Table
import os
import math
import random
import copy

from AUTOGRAD import Utils, Modules, Optimizers
from AUTOGRAD import data_procs_engine

import sys
sys.setrecursionlimit(10000) # this is needed for proper graph generation, will fix it next ver.

def train_and_plot(model, X_train, y_train, X_val, y_val, args):
    """Trains the model and plots the train/val loss and accuracy curves."""
    train_losses = []
    val_losses = []
    val_loss = None 
    
    # Early Stopping & Safety logic
    best_val_loss = float('inf')
    trigger_times = 0
    best_weights = None
    stopped_epoch = None
    k = 0  
    
    params = model.parameters()
    lr = args.lr
    
    if args.opt == "adam":
        optimizer = Optimizers.Adam(params, learning_rate=lr)
    elif args.opt == "adamw":
        optimizer = Optimizers.AdamW(params, learning_rate=lr, weight_decay=args.weight_decay)
    elif args.opt == "rmsprop":
        optimizer = Optimizers.RMSprop(params, learning_rate=lr)
    else:
        optimizer = Optimizers.SGD(params, learning_rate=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    metric_name = "MAE" if args.loss in ["mse", "l1"] else "Accuracy"
    
    table = Table(title=f"Training ({args.model.upper()} | {args.loss.upper()} | {args.opt.upper()})")
    table.add_column("Epoch", justify="right", style="cyan")
    table.add_column("Train Loss", justify="right", style="blue")
    table.add_column("Val Loss", justify="right", style="magenta")
    table.add_column(f"Val {metric_name}", justify="right", style="green")

    def _run_pass(xs, ys, is_training=False):
        ypred = []
        for x in xs:
            out = model(x)
            if args.loss == "cce":
                ypred.append(out)
            else:
                ypred.append(out[0] if isinstance(out, list) else out)
                
        if args.loss == "mse": loss = Utils.mse_loss(ypred, ys)
        elif args.loss == "bce": loss = Utils.bce_loss(ypred, ys)
        elif args.loss == "hinge": loss = Utils.hinge_loss(ypred, ys)
        elif args.loss == "l1": loss = Utils.l1_loss(ypred, ys)
        elif args.loss == "cce": loss = Utils.categorical_cross_entropy(ypred, ys)
            
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if args.loss in ["mse", "l1"]:
            current_metric = sum(abs(p.data - y) for p, y in zip(ypred, ys)) / len(ys)
            metric_str = f"{current_metric:.4f}"
        else:
            correct = 0
            for p, y in zip(ypred, ys):
                if args.loss == "hinge":
                    pred_label = 1.0 if p.data >= 0 else -1.0
                    if pred_label == y: correct += 1
                elif args.loss == "cce":
                    pred_class = max(range(len(p)), key=lambda i: p[i].data)
                    if pred_class == int(y): correct += 1
                else:
                    if p.data >= 0:
                        prob = 1.0 / (1.0 + math.exp(-p.data))
                    else:
                        z = math.exp(p.data) 
                        prob = z / (1.0 + z)
                    if (prob >= args.threshold) == (y >= 0.5): correct += 1
                
            current_metric = (correct / len(ys)) * 100
            metric_str = f"{current_metric:.1f}%"
            
        return loss, metric_str

    print(f"Data Split: {len(X_train)} Train | {len(X_val)} Validation")

    try:
        with Live(table, refresh_per_second=10):
            for k in range(args.epochs):
                
                # dynamic per-epoch shuffling
                if args.shuffle_per_epoch:
                    train_data = list(zip(X_train, y_train))
                    random.shuffle(train_data)
                    shuffled_X, shuffled_y = zip(*train_data)
                    X_train, y_train = list(shuffled_X), list(shuffled_y)

                # training
                train_loss, _ = _run_pass(X_train, y_train, is_training=True)
                train_losses.append(train_loss.data)
                
                # validation
                val_loss, val_metric_str = _run_pass(X_val, y_val, is_training=False)
                val_losses.append(val_loss.data)
                
                # update ui
                if k % 10 == 0 or k == args.epochs - 1:
                    table.add_row(f"{k}", f"{train_loss.data:.4f}", f"{val_loss.data:.4f}", val_metric_str)
                    
                # early stopping logic
                min_delta = 1e-4
                if args.patience > 0:
                    if val_loss.data < (best_val_loss - min_delta):
                        best_val_loss = val_loss.data
                        trigger_times = 0
                        best_weights = copy.deepcopy(model.state_dict()) 
                    else:
                        trigger_times += 1
                        
                    if trigger_times >= args.patience:
                        stopped_epoch = k
                        table.add_row(f"{k} (STOP)", f"{train_loss.data:.4f}", f"{val_loss.data:.4f}", val_metric_str, style="bold red")
                        break

    except KeyboardInterrupt:
        print("\n\n[!] Training manually interrupted via Ctrl+C!")
        stopped_epoch = k

    # ALWAYS rewind to best weights if early stopping was active
    if best_weights is not None:
        status_msg = f"at epoch {stopped_epoch}" if stopped_epoch is not None else "at end of run"
        print(f"\nRewinding model weights to best epoch {status_msg} (Val Loss: {best_val_loss:.4f})")
        model.load_state_dict(best_weights)
        
    if args.visualize and len(train_losses) > 0:
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, color='blue', label='Train Loss', linewidth=2, marker='o', markersize=3)
        plt.plot(val_losses, color='orange', label='Val Loss', linewidth=2, marker='x', markersize=3)
        
        title_suffix = f" (Stopped at {stopped_epoch})" if stopped_epoch is not None else ""
        plt.title(f"Training Curves ({args.model.upper()} | {args.loss.upper()} | LR={lr}){title_suffix}")
        
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        save_dir = os.path.expanduser("Model")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'training.svg')
        plt.savefig(save_path, format="svg", bbox_inches="tight", pad_inches=0.1)
        plt.close()
        print(f"Graph saved to {save_path}")

    if args.save_weights:
        model.save(filename=args.save_weights)
        print(f"Model weights successfully saved to {args.save_weights}")

    if val_loss is not None:
        Utils.export_telemetry(val_loss, filename="telemetry.json")
        
    return train_losses, val_losses


def main():
    parser = argparse.ArgumentParser(description="The AutoGrad Engine")
    
    # training loop args
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV dataset")
    parser.add_argument("--visualize", action="store_true", help="Plot and save loss curve")
    parser.add_argument("--save_weights", type=str, default=None, help="Save model weights")
    parser.add_argument("--load_weights", type=str, default=None, help="Path to load weights")
    parser.add_argument("--patience", type=int, default=10, help="Epochs to wait for improvement before stopping (0 to disable)")
    
    # data management args
    parser.add_argument("--shuffle_initial", action="store_true", help="Shuffle dataset once upon loading via data_procs_engine")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of data for training (e.g. 0.8 = 80%% train)")
    parser.add_argument("--shuffle_per_epoch", action="store_true", help="Dynamically reshuffle training set every epoch")
    
    # architecture args
    parser.add_argument("--model", type=str, choices=["mlp", "logistic", "linear"], default="mlp", help="Model architecture")
    parser.add_argument("--hidden", type=int, nargs="+", default=[4], help="Hidden layer sizes for MLP (e.g. --hidden 4 4)")
    
    # optimizer & loss args
    parser.add_argument("--opt", type=str, choices=["sgd", "adam", "adamw", "rmsprop"], default="adam", help="Optimizer choice")
    parser.add_argument("--loss", type=str, choices=["mse", "bce", "hinge", "l1", "cce"], default="bce", help="Loss function")
    parser.add_argument("--classes", type=int, default=3, help="Number of classes (only used for CCE)")
    parser.add_argument("--momentum", type=float, default=0.0, help="Momentum for SGD")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="L2 Penalty for SGD/AdamW")

    # args to specify probability threshold in classification model.
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold (0.0 to 1.0)")
    args = parser.parse_args()
    
    print("========== The Autograd Engine ==========")
    
    # 1. Delegate loading and initial shuffling to data_procs_engine
    xs, ys = data_procs_engine.load_and_prep_data(args.data, shuffle_initial=args.shuffle_initial)

    # 2. Preprocess targets if needed
    if args.loss == "hinge":
        ys = [1.0 if y > 0.5 else -1.0 for y in ys]
        print("Remapped targets to +1 / -1 for Hinge Loss.")
        
    # 3. Apply Train/Val Split in main based on args.train_ratio
    split_idx = int(len(xs) * args.train_ratio)
    X_train, y_train = xs[:split_idx], ys[:split_idx]
    
    if len(xs) - split_idx > 0:
        X_val, y_val = xs[split_idx:], ys[split_idx:]
    else:
        X_val, y_val = X_train, y_train
        print("Warning: Dataset too small for validation split. Using train data for validation.")
        
    input_features = len(xs[0])
    
    if args.model == "logistic":
        print(f"Initializing Logistic Regression ({input_features} inputs)")
        model = Modules.LogisticRegression(input_features)
    elif args.model == "linear":
        print(f"Initializing Linear Regression ({input_features} inputs)")
        model = Modules.LinearRegression(input_features)
    else:
        output_size = args.classes if args.loss == "cce" else 1
        layer_sizes = args.hidden + [output_size]
        print(f"Initializing MLP with layers: [{input_features}] -> {layer_sizes}")
        model = Modules.MLP(input_features, layer_sizes)

    if args.load_weights:
        print(f"##Loading weights from {args.load_weights}##")
        model.load(filename=args.load_weights)
    
    # 4. Train using split data
    train_and_plot(model, X_train, y_train, X_val, y_val, args)

if __name__ == "__main__":
    main()