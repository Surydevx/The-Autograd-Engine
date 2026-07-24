import math
from Engine import Value

def test_addition_and_multiplication():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    d = a * b  
    
    c.backward()
    assert c.data == 5.0
    assert a.gradient == 1.0
    assert b.gradient == 1.0
    
    # Resetting gradients
    a.gradient, b.gradient = 0.0, 0.0
    d.backward()
    assert d.data == 6.0
    assert a.gradient == 3.0
    assert b.gradient == 2.0

def test_complex_expression():
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    
    assert y.data == -20.0
    assert x.gradient == 46.0

def test_sigmoid():
    x = Value(0.0)
    y = x.sigmoid()
    y.backward()
    
    # Sigmoid(0) = 0.5
    assert y.data == 0.5
    # Derivative of Sigmoid at 0 is 0.25
    assert x.gradient == 0.25

def test_log_and_bce_components():
    p = Value(0.5)
    y_true = Value(1.0)
    
    # Simulating BCE loss term: y * log(p)
    loss_term = y_true * p.log()
    loss_term.backward()
    
    assert math.isclose(loss_term.data, math.log(0.5))
    assert p.gradient == 2.0  # (1/0.5) * 1.0 = 2.0