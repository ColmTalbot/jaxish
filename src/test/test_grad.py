import numpy as np
import pytest
from array_api_compat import array_namespace

from .. import grad


def test_grad_numpy_raises(undiffable):
    def f(x):
        return undiffable.sum(x**2)

    g = grad(f)
    x = undiffable.array([1.0, 2.0, 3.0])
    with pytest.raises(NotImplementedError):
        g(x)


def test_grad_jax_mlx(diffable):
    arr = diffable.array([1.0, 2.0, 3.0])

    def f(x):
        return diffable.sum(x**2)

    g = grad(f)
    grad_val = g(arr)

    # The gradient of sum(x**2) is 2*x
    expected = 2 * np.array([1.0, 2.0, 3.0])
    expected = np.asarray(expected)
    assert np.allclose(grad_val, expected)


def test_jit_with_float_argument_default_backend_fails():
    @grad
    def f(x):
        return np.sum(x**2)

    with pytest.raises(NotImplementedError):
        f(3.0)


def test_jit_with_float_argument_and_explicit_backend(diffable):
    @grad
    def f(x):
        return diffable.sum(x**2)

    result = f(diffable.array(3.0), xp=diffable)
    assert result == 6.0
    assert array_namespace(result) == diffable
