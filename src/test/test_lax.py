import array_api_compat
import numpy as np
import pytest

from ..lax import cond, scan, select, while_loop


def test_scan_carry_array(xp):
    def f(carry, x):
        return carry + x, carry * x

    carry = xp.array(1.0)
    xs = xp.array([2.0, 3.0, 4.0])
    final_carry, ys = scan(f, carry, xs)
    assert np.allclose(final_carry, 1.0 + 2.0 + 3.0 + 4.0)
    assert np.allclose(ys, [1.0 * 2.0, 3.0 * 3.0, 6.0 * 4.0])
    assert array_api_compat.is_array_api_obj(final_carry)
    assert array_api_compat.is_array_api_obj(ys)
    assert array_api_compat.array_namespace(final_carry) == xp
    assert array_api_compat.array_namespace(ys) == xp


def test_scan_carry_float():
    def f(carry, x):
        return carry + x, carry * x

    carry = 1.0
    xs = [2.0, 3.0, 4.0]
    final_carry, ys = scan(f, carry, xs)
    assert np.allclose(final_carry, 1.0 + 2.0 + 3.0 + 4.0)
    assert np.allclose(ys, [1.0 * 2.0, 3.0 * 3.0, 6.0 * 4.0])


def test_scan_carry_list_of_arrays(xp):
    def f(carry, x):
        a, b = carry
        return [a + x, b - x], a * b

    carry = [xp.array(1.0), xp.array(10.0)]
    xs = xp.array([2.0, 3.0, 4.0])
    final_carry, ys = scan(f, carry, xs)
    assert np.allclose(final_carry[0], 1.0 + 2.0 + 3.0 + 4.0)
    assert np.allclose(final_carry[1], 10.0 - 2.0 - 3.0 - 4.0)
    assert ys.shape[0] == 3
    assert array_api_compat.is_array_api_obj(final_carry[0])
    assert array_api_compat.is_array_api_obj(ys)
    assert array_api_compat.array_namespace(final_carry[0]) == xp
    assert array_api_compat.array_namespace(ys) == xp


def test_scan_carry_list_of_floats_raises():
    def f(carry, x):
        a, b = carry
        return [a + x, b - x], a * b

    carry = [1.0, 10.0]
    xs = [2.0, 3.0, 4.0]
    with pytest.raises(TypeError):
        scan(f, carry, xs)


def test_scan_xs_none_length_given(xp):
    def f(carry, _):
        return carry + 1, carry

    carry = xp.array(0)
    length = 4
    final_carry, ys = scan(f, carry, xs=None, length=length)
    assert final_carry == 4
    assert np.allclose(ys, [0, 1, 2, 3])
    assert array_api_compat.array_namespace(final_carry) == xp
    assert array_api_compat.array_namespace(ys) == xp


def test_cond(xp):
    def true_fun(x):
        return xp.array(x + 1)

    def false_fun(x):
        return xp.array(x - 1)

    x = xp.array([5.0, 6.0, 7.0])
    pred = True
    result = cond(pred, true_fun, false_fun, x)
    assert np.allclose(result, x + 1)
    assert array_api_compat.array_namespace(result) == xp

    pred = False
    result = cond(pred, true_fun, false_fun, x)
    assert np.allclose(result, x - 1)
    assert array_api_compat.array_namespace(result) == xp


def test_select(xp):
    x = xp.array([5.0, 6.0, 7.0])
    pred = x > 6.5
    on_true = xp.array([1.0, 2.0, 3.0])
    on_false = xp.array([4.0, 5.0, 6.0])
    result = select(pred, on_true, on_false)
    assert np.allclose(result, [4.0, 5.0, 3.0])
    assert array_api_compat.array_namespace(result) == xp


def test_while_loop(xp):
    def cond_fun(val):
        return val < 5

    def body_fun(val):
        return val + 1

    val = xp.array(0)
    result = while_loop(cond_fun, body_fun, val)
    assert result == xp.array(5)
    assert array_api_compat.array_namespace(result) == xp
