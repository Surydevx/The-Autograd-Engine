class value:
    def __init__(self, data, _prev=(), _op=""):
        self.data = float(data)

        self._prev = set(_prev)

        self._op = _op

        self.grad = 0.0

        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        if isinstance(other, value):
            out_data = self.data + other.data
            out_object = value(out_data, (self, other), "+")

            def _backward():
                self.grad += 1.0 * out_object.grad
                other.grad += 1.0 * out_object.grad

            out_object._backward = _backward
            return out_object
        else:
            other = value(other)
            out_data = self.data + other.data
            out_object = value(out_data, (self, other), "+")

            def _backward():
                self.grad += 1.0 * out_object.grad
                other.grad += 1.0 * out_object.grad

            out_object._backward = _backward
            return out_object

    def __mul__(self, other):
        if isinstance(other, value):
            out_data = self.data * other.data
            out_object = value(out_data, (self, other), "*")

            def _backward():
                self.grad += other.data * out_object.grad
                other.grad += self.data * out_object.grad

            out_object._backward = _backward
            return out_object
        else:
            other = value(other)
            out_data = self.data * other.data
            out_object = value(out_data, (self, other), "*")

            def _backward():
                self.grad += other.data * out_object.grad
                other.grad += self.data * out_object.grad

            out_object._backward = _backward
            return out_object

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for x in v._prev:
                    build_topo(x)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


# --- THE AUTOMATED AUTOGRAD TEST ---
if __name__ == "__main__":
    a = value(2.0)
    b = value(3.0)

    # Forward Pass
    c = a * b
    d = c + a

    # Automated Backward Pass
    d.backward()

    print(f"Mathematical output of d: {d.data}")
    print(f"Gradient of a: {a.grad}")
    print(f"Gradient of b: {b.grad}")

