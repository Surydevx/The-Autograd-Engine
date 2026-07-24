import math
class Value:
    """ It stores a single scalar value and its gradient """
    def __init__(self, data, _previous=(), _operation=''):
        self.data = float(data)
        self._previous = set(_previous)
        self._operation = _operation
        self.gradient = 0.0
        self._backward = lambda: None

    def __repr__(self):
        return f'Value(data={self.data}, grad={self.gradient})'

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output_object = Value(self.data + other.data, _previous=(self, other), _operation='+')
        
        def _backward():
            self.gradient += 1.0 * output_object.gradient
            other.gradient += 1.0 * output_object.gradient
            
        output_object._backward = _backward
        return output_object

    def __radd__(self, other): # to satsify the commutativity in real numbers under addition.
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output_object = Value(self.data * other.data, _previous=(self, other), _operation="*")
        
        def _backward():
            self.gradient += other.data * output_object.gradient
            other.gradient += self.data * output_object.gradient
            
        output_object._backward = _backward
        return output_object

    def __rmul__(self, other):
        return self * other 

    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        try:
            output_data = self.data ** other.data
            if isinstance(output_data, complex):
                output_data = float('nan')
        except (ZeroDivisionError, ValueError):
            output_data = float('nan') 

        output_object = Value(output_data, _previous=(self, other), _operation=f'**{other.data}')

        def _backward():
            if self.data == 0 and (other.data - 1) < 0:
                pass 
            else:
                grad_val = other.data * (self.data ** (other.data - 1))
                if isinstance(grad_val, complex):
                    grad_val = grad_val.real
                self.gradient += grad_val * output_object.gradient
                
            if self.data > 0:
                other_grad = (self.data ** other.data) * math.log(self.data)
                if isinstance(other_grad, complex):
                    other_grad = other_grad.real
                other.gradient += other_grad * output_object.gradient

        output_object._backward = _backward
        return output_object

    def exp(self): # exponential functiona as e^x
        x = max(min(self.data, 700), -700)    # prevents OverFlowError    
        output_data = math.exp(x)
        output_object = Value(output_data, (self,), "exp")

        def _backward():
            self.gradient += output_object.data * output_object.gradient

        output_object._backward = _backward
        return output_object

    def sin(self): # sine function
        x = self.data
        output_data = math.sin(x)
        output_object = Value(output_data, (self,), "sin")

        def _backward():
            self.gradient += math.cos(x) * output_object.gradient

        output_object._backward = _backward
        return output_object

    def cos(self):
        x = self.data
        output_data = math.cos(x)
        output_object = Value(output_data, (self,), "cos")

        def _backward():
            self.gradient += (-math.sin(x)) * output_object.gradient

        output_object._backward = _backward
        return output_object

    def tanh(self):
        x = self.data
        t = math.tanh(x)
        output_object = Value(t, (self,), "tanh")

        def _backward():
            self.gradient += (1.0 - t**2) * output_object.gradient

        output_object._backward = _backward
        return output_object

    def log(self):
        x = self.data
        epsilon = 1e-8
        output_data = math.log(x + epsilon if x >= 0 else epsilon)
        
        output_object = Value(output_data, (self,), "log")
    
        def _backward():
            self.gradient += (1.0 / (x + epsilon)) * output_object.gradient

        output_object._backward = _backward
        return output_object
        
    
    def sigmoid(self):
        x = self.data
        if x >= 0:
            output_data = 1.0 / (1.0 + math.exp(-x))
        else:
            exp_x = math.exp(x)
            output_data = exp_x / (1.0 + exp_x)
            
        output_object = Value(output_data, (self,), "sigmoid")

        def _backward():
            local_gradient = output_object.data * (1.0 - output_object.data)
            self.gradient += local_gradient * output_object.gradient

        output_object._backward = _backward
        return output_object


    def relu(self, leak=0.0):
        output_data = self.data if self.data > 0 else leak * self.data
        output_object = Value(output_data, _previous=(self,), _operation='ReLU')

        def _backward():
            local_gradient = 1.0 if self.data > 0 else leak
            self.gradient += local_gradient * output_object.gradient
                
        output_object._backward = _backward
        return output_object

    def backward(self):
        topo = []
        visited = set()
        stack = [(self, False)]
        while stack:
            v, is_expanded = stack.pop()
            if not is_expanded:
                if v not in visited:
                    visited.add(v)
                    stack.append((v, True))
                    for child in v._previous:
                        stack.append((child, False))
            else:
                topo.append(v)
        self.gradient = 1.0
        for v in reversed(topo):
            v._backward()

# some additional operations.
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)
# := done says, walrus