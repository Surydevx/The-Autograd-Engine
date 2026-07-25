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
    
    # Early Stopping logic
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
    
    metric_name = "MAE" if args.loss in ["mse", "l1"] else "Accuracy"
    
    table = Table(title=f"Training ({args.model.upper()} | {args.loss.upper()} | {args.opt.upper()})")
    table.add_column("Epoch", justify="right", style="cyan")
    table.add_column("Loss", justify="right", style="magenta")
    table.add_column(metric_name, justify="right", style="green")

    with Live(table, refresh_per_second=10):
        for k in range(args.epochs):
            ypred = []
            for x in xs:
                out = model(x)
                # CCE requires the full list of output logits, other loss functions just need the scalar
                if args.loss == "cce":
                    ypred.append(out)
                else:
                    ypred.append(out[0] if isinstance(out, list) else out)
                
            if args.loss == "mse": loss = Utils.mse_loss(ypred, ys)
            elif args.loss == "bce": loss = Utils.bce_loss(ypred, ys)
            elif args.loss == "hinge": loss = Utils.hinge_loss(ypred, ys)
            elif args.loss == "l1": loss = Utils.l1_loss(ypred, ys)
            elif args.loss == "cce": loss = Utils.categorical_cross_entropy(ypred, ys)
                
            losses.append(loss.data)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if args.loss in ["mse", "l1"]:
                # calculate mean absolute arror
                current_metric = sum(abs(p.data - y) for p, y in zip(ypred, ys)) / len(ys)
                metric_str = f"{current_metric:.4f}"
            else:
                # calculate accuracy
                correct = 0
                for p, y in zip(ypred, ys):
                    if args.loss == "hinge":
                        if (p.data > 0) == (y > 0): correct += 1
                    elif args.loss == "cce":
                        pred_class = max(range(len(p)), key=lambda i: p[i].data)
                        if pred_class == int(y): correct += 1
                    else:
                        # BCE
                        if p.data >= 0:
                            prob = 1.0 / (1.0 + math.exp(-p.data))
                        else:
                            z = math.exp(p.data) 
                            prob = z / (1.0 + z)
                        if (prob >= args.threshold) == (y >= 0.5): correct += 1
                        
                current_metric = (correct / len(ys)) * 100
                metric_str = f"{current_metric:.1f}%"
            
            if k % 10 == 0 or k == args.epochs - 1:
                table.add_row(f"{k}", f"{loss.data:.4f}", metric_str)
                
            min_delta = 1e-4
            
            if args.patience > 0:
                if loss.data < (best_loss - min_delta):
                    best_loss = loss.data
                    trigger_times = 0
                    best_weights = model.state_dict() 
                else:
                    trigger_times += 1
                    
                if trigger_times >= args.patience:
                    stopped_epoch = k
                    table.add_row(f"{k} (STOP)", f"{loss.data:.4f}", metric_str, style="bold red")
                    break

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
        
        save_dir = os.path.expanduser("Model")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'training.svg')
        plt.savefig(save_path, format ="svg", bbox_inches = "tight", pad_inches=0.1)
        plt.close()
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
    parser.add_argument("--loss", type=str, choices=["mse", "bce", "hinge", "l1", "cce"], default="bce", help="Loss function")
    parser.add_argument("--classes", type=int, default=3, help="Number of classes (only used for CCE)")
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

    if args.loss == "hinge":
        ys = [1.0 if y > 0.5 else -1.0 for y in ys]
        print("Remapped targets to +1 / -1 for Hinge Loss.")
        
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
    
    # 3. Train
    train_and_plot(model, xs, ys, args)

if __name__ == "__main__":
    main()