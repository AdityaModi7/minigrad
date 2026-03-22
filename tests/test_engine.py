"""Tests for the MiniGrad autograd engine — verifies gradients against
manually computed values and checks all operations."""

from minigrad.engine import Value


def test_basic_ops():
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = a * b
    c.backward()
    assert a.grad == -3.0  # dc/da = b
    assert b.grad == 2.0   # dc/db = a


def test_add_and_mul():
    x = Value(2.0)
    y = Value(3.0)
    z = x * y + x
    z.backward()
    assert x.grad == 4.0   # dz/dx = y + 1 = 4
    assert y.grad == 2.0   # dz/dy = x = 2


def test_relu():
    a = Value(-2.0)
    b = Value(3.0)
    c = a.relu()
    d = b.relu()
    c.backward()
    d.backward()
    assert c.data == 0.0
    assert d.data == 3.0
    assert a.grad == 0.0   # relu kills gradient for negative
    assert b.grad == 1.0   # relu passes gradient for positive


def test_power():
    x = Value(3.0)
    y = x ** 2
    y.backward()
    assert y.data == 9.0
    assert x.grad == 6.0   # dy/dx = 2x = 6


def test_division():
    a = Value(6.0)
    b = Value(3.0)
    c = a / b
    c.backward()
    assert abs(c.data - 2.0) < 1e-6
    assert abs(a.grad - 1 / 3) < 1e-6
    assert abs(b.grad - (-6 / 9)) < 1e-6


def test_tanh():
    import math
    x = Value(0.0)
    y = x.tanh()
    y.backward()
    assert abs(y.data - 0.0) < 1e-6
    assert abs(x.grad - 1.0) < 1e-6  # tanh'(0) = 1


def test_complex_expression():
    """Test a more complex expression matches hand-computed gradients."""
    a = Value(2.0)
    b = Value(3.0)
    c = a + b         # 5
    d = a * b         # 6
    e = c * d         # 30
    e.backward()
    # de/da = d + c*b = 6 + 5*3 = 21
    assert a.grad == 21.0
    # de/db = d + c*a = 6 + 5*2 = 16  ... wait
    # e = (a+b)*(a*b), de/db = a*b + (a+b)*a = 6 + 5*2 = 16
    # Actually: de/db via chain rule:
    # de/dc * dc/db + de/dd * dd/db = d*1 + c*a = 6 + 10 = 16
    assert b.grad == 16.0


def test_gradient_accumulation():
    """When a value is used multiple times, gradients should accumulate."""
    a = Value(3.0)
    b = a + a  # b = 2a
    b.backward()
    assert a.grad == 2.0


def test_no_grad_on_fresh_value():
    v = Value(42.0)
    assert v.grad == 0.0


if __name__ == '__main__':
    test_basic_ops()
    test_add_and_mul()
    test_relu()
    test_power()
    test_division()
    test_tanh()
    test_complex_expression()
    test_gradient_accumulation()
    test_no_grad_on_fresh_value()
    print("All tests passed!")
