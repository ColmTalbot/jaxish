from unittest import mock

import numpy as np
import pytest
from array_api_compat import array_namespace, get_namespace

from .. import jit


def test_jit_calls_backend_compile_with_decorator(xp):
    @jit
    def f(x):
        return (x**2).sum()

    _generic_calls_compile(xp, f)


def test_jit_calls_backend_compile_with_function(xp):
    def f(x):
        return (x**2).sum()

    f = jit(f)
    _generic_calls_compile(xp, f)


def _generic_calls_compile(xp, f):
    arr = xp.array([1.0, 2.0, 3.0])
    match get_namespace(arr).__name__:
        case "jax.numpy":
            import jax.numpy as xp

            target = "jax.jit"
        case "mlx.core":
            import mlx.core as xp

            target = "mlx.core.compile"
        case _:
            pytest.skip(f"Backend {xp.__name__} does not support compilation")

    with mock.patch(target) as mock_jit:
        arr = xp.array([1.0, 2.0, 3.0])
        f(arr)
        assert mock_jit.called, f"{target} was not called"


def test_jit_returns_correct_value(xp):
    @jit
    def f(x):
        return (x**2).sum()

    arr = xp.array([1.0, 2.0, 3.0])
    result = f(arr)
    expected = (arr**2).sum()
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_jit_with_float_argument_default_backend():
    @jit
    def f(x):
        return x * x

    result = f(3.0)
    assert result == 9.0
    assert isinstance(result, float)


def test_jit_with_float_argument_and_explicit_backend(xp):
    @jit
    def f(x):
        return x * x

    result = f(xp.array(3.0), xp=xp)
    assert result == 9.0
    assert array_namespace(result) == xp
