import argparse
import matplotlib.pyplot as plt
from rich.live import Live
from rich.table import Table
import os
import math

from AUTOGRAD import Utils, Modules, Optimizers


def train_and_plot(model, xs, ys, args):
    """Trains the model and plots the loss/accuracy curves."""
    losses = []
    loss = None
    
    # --- Early Stopping Trackers ---
    best_loss = float('inf')
    trigger_times = 0
    best_weights = None
    stopped_epoch = None
    
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
    
    table = Table(title=f"Training ({args.model.upper()} | {args.loss.upper()} | {args.opt.upper()})")
    table.add_column("Epoch", justify="right", style="cyan")
    table.add_column("Loss", justify="right", style="magenta")
    table.add_column("Accuracy", justify="right", style="green")

    with Live(table, refresh_per_second=10):
        for k in range(args.epochs):
            ypred = []
            for x in xs:
                out = model(x)
                ypred.append(out[0] if isinstance(out, list) else out)
                
            if args.loss == "mse": loss = Utils.mse_loss(ypred, ys)
            elif args.loss == "bce": loss = Utils.bce_loss(ypred, ys)
            elif args.loss == "hinge": loss = Utils.hinge_loss(ypred, ys)
            elif args.loss == "l1": loss = Utils.l1_loss(ypred, ys)
                
            losses.append(loss.data)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            correct = 0
            for p, y in zip(ypred, ys):
                if p.data >= 0:
                    prob = 1.0 / (1.0 + math.exp(-p.data))
                else:
                    z = math.exp(p.data) 
                    prob = z / (1.0 + z)
    
                if (prob >= args.threshold) == (y >= 0.5):
                    correct += 1
                    
            accuracy = (correct / len(ys)) * 100
            
            if k % 10 == 0 or k == args.epochs - 1:
                table.add_row(f"{k}", f"{loss.data:.4f}", f"{accuracy:.1f}%")
                
            # --- Early Stopping Logic ---
            min_delta = 1e-4  # 0.0001
            
            if args.patience > 0:
                # FIX: Must improve by at least min_delta!
                if loss.data < (best_loss - min_delta):
                    best_loss = loss.data
                    trigger_times = 0
                    best_weights = model.state_dict() 
                else:
                    trigger_times += 1
                    
                if trigger_times >= args.patience:
                    stopped_epoch = k
                    table.add_row(f"{k} (STOP)", f"{loss.data:.4f}", f"{accuracy:.1f}%", style="bold red")
                    break

    # Once the table is finished drawing, print the early stopping result
    if stopped_epoch is not None:
        print(f"\n Early stopping triggered at epoch {stopped_epoch}!")
        print(f"Rewinding model weights to best epoch (Loss: {best_loss:.4f})")
        model.load_state_dict(best_weights)
        
    if args.visualize and len(losses) > 0:
        plt.figure(figsize=(8, 5))
        plt.plot(losses, color='blue', linewidth=2, marker='o', markersize=4)
        
        title_suffix = f" (Stopped at {stopped_epoch})" if stopped_epoch else ""
        plt.title(f"Training Loss ({args.model.upper()} | {args.loss.upper()} | LR={lr}){title_suffix}")
        
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        save_dir = os.path.expanduser("~/Model")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'training_loss.png')
        plt.savefig(save_path)
        print(f"Graph saved to {save_path}")

    if args.save_weights:
        model.save(filename=args.save_weights)
        print(f"Model weights successfully saved to {args.save_weights}")

    if loss is not None:
        Utils.export_telemetry(loss, filename="telemetry.json")
        
    return losses


def main():
    parser = argparse.ArgumentParser(description="The Autograd Engine")
    
    # Training Loop Args
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV dataset")
    parser.add_argument("--visualize", action="store_true", help="Plot and save loss curve")
    parser.add_argument("--save_weights", type = str,default = None , help="Save model weights")
    parser.add_argument("--load_weights", type=str, default = None, help="Path to load weights")
    parser.add_argument("--patience", type=int, default=10, help="Epochs to wait for improvement before stopping (0 to disable)")
    # Architecture Args
    parser.add_argument("--model", type=str, choices=["mlp", "logistic", "linear"], default="mlp", help="Model architecture")
    parser.add_argument("--hidden", type=int, nargs="+", default=[4], help="Hidden layer sizes for MLP (e.g. --hidden 4 4)")
    
    # Optimizer & Loss Args
    parser.add_argument("--opt", type=str, choices=["sgd", "adam", "adamw", "rmsprop"], default="adam", help="Optimizer choice")
    parser.add_argument("--loss", type=str, choices=["mse", "bce", "hinge", "l1"], default="bce", help="Loss function")
    parser.add_argument("--momentum", type=float, default=0.0, help="Momentum for SGD")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="L2 Penalty for SGD/AdamW")

    # Args to specify probability threshold in classification systems.
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold (0.0 to 1.0)")
    args = parser.parse_args()
    
    print("========== The Autograd Engine ==========")
    
    xs, ys = None, None
    if args.data:
        xs, ys = Utils.load_csv(args.data)
        
    if not xs or not ys:
        print("Using default logic gate dataset (OR Gate)...")
        xs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
        ys = [0.0, 1.0, 1.0, 1.0]
        
    input_features = len(xs[0])
    
    if args.model == "logistic":
        print(f"Initializing Logistic Regression ({input_features} inputs)")
        model = Modules.LogisticRegression(input_features)
    elif args.model == "linear":
        print(f"Initializing Logistic Regression ({input_features} inputs)")
        model = Modules.LinearRegression(input_features)
    else:
        layer_sizes = args.hidden + [1]
        print(f"Initializing MLP with layers: [{input_features}] -> {layer_sizes}")
        model = Modules.MLP(input_features, layer_sizes)

    if args.load_weights:
            print(f"##Loading weights from {args.load_weights}##")
            model.load(filename=args.load_weights)
    
    # 3. Train
    train_and_plot(model, xs, ys, args)

if __name__ == "__main__":
    main()

### phewwwwwwwwwwwwww, i need a break *_*