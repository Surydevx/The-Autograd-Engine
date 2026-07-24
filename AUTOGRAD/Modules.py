import random
import json
import os
from AUTOGRAD.Engine import Value

path_name = "Model"

class Module:

    """
    standard utility methods.
    """
    def zero_grad(self):
        """this function resets the gradients of all parameters to zero."""
        for p in self.parameters():
            p.gradient = 0.0

    def parameters(self):
        """ outputs a list of all trainable Value parameters."""
        return []

    def state_dict(self):
        """outputs a list of float values"""
        return [p.data for p in self.parameters()]

    def load_state_dict(self, state):
        """Restores raw float values into the model parameters."""
        params = self.parameters()
        if len(params) != len(state):
            raise ValueError(f"State mismatch: model expects {len(params)} values, got {len(state)}.")
        for p, val in zip(params, state):
            p.data = float(val)

    def save(self, filename="model_weights.json"):
        """Saves current state_dict to a JSON file."""

        save_path = os.path.expanduser(path_name)
        os.makedirs(save_path, exist_ok=True)    
        save_path = os.path.join(save_path, filename)

        with open(save_path, 'w') as f:
            json.dump(self.state_dict(), f, indent=4)
        print(f"Saved model parameters to {filename}")

    def load(self, filename="model_weights.json"):
        """Loads state_dict from a JSON file."""

        load_path = os.path.expanduser(path_name)
        load_path = os.path.join(load_path, filename)
        if not os.path.isfile(load_path):
            print(f"Warning: Cannot load weights. '{load_path}' does not exist. Keeping random initialization.")
            return        
        with open(load_path, 'r') as f:
            state = json.load(f)
        self.load_state_dict(state)
        print(f"Loaded model parameters from {filename}")


class Neuron(Module):
    def __init__(self, n_in, activation='relu'):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(random.uniform(-1, 1))
        self.activation = activation

    def __call__(self, x):
        x = [x] if isinstance(x, (int, float, Value)) else x
        if len(x) != len(self.w):
            raise ValueError(f"Shape mismatch: Expected {len(self.w)} inputs, got {len(x)}")
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        if self.activation == 'relu':
            return act.relu()
        elif self.activation == 'sigmoid':
            return act.sigmoid()
        elif self.activation == 'tanh':
            return act.tanh()
        else:
            return act

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    """A fully-connected layer consisting of multiple Neurons."""
    def __init__(self, n_in, n_out, **kwargs):
        self.neurons = [Neuron(n_in, **kwargs) for _ in range(n_out)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP(Module):
    """Multi-Layer Perceptron"""
    def __init__(self, n_in, n_outs):
        n_outs = [n_outs] if isinstance(n_outs, int) else n_outs
        sizes = [n_in] + n_outs
        
        self.layers = []
        for i in range(len(n_outs)):
            act = 'relu' if i != len(n_outs) - 1 else 'linear'
            self.layers.append(Layer(sizes[i], sizes[i+1], activation=act))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


class LinearRegression(Module):
    """
    Linear Regression: y = w1*x1 + w2*x2 + ... + b 
    """
    def __init__(self, n_in):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)

    def __call__(self, x):
        x = [x] if isinstance(x, (int, float, Value)) else x
        if len(x) != len(self.w):
            raise ValueError(f"Shape mismatch: Expected {len(self.w)} inputs, got {len(x)}")
        return sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

    def parameters(self):
        return self.w + [self.b]

class LogisticRegression(Module):
    """
    A Logistic Regression model for binary classification.
    """
    def __init__(self, n_in):
        # Initialize weights randomly and bias to 0
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)

    def __call__(self, x):
        x = [x] if isinstance(x, (int, float, Value)) else x
        if len(x) != len(self.w):
            raise ValueError(f"Shape mismatch: Expected {len(self.w)} inputs, got {len(x)}")
        logit = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)        
        return logit # it returns raw logit, in bce_loss under utils.py we would impose sigmoid function on it.

    def parameters(self):
        return self.w + [self.b]