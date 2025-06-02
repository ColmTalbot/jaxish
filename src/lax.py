import array_api_compat
import numpy as np

__all__ = ["cond", "select", "scan", "while_loop"]


def _scan(f, init, xs, length=None, *, xp):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, xp.stack(ys)


def _while_loop(cond_fun, body_fun, init_val):
    val = init_val
    while cond_fun(val):
        val = body_fun(val)
    return val


def cond(pred, true_fun, false_fun, *operands):
    if array_api_compat.is_jax_namespace(array_api_compat.array_namespace(*operands)):
        from jax import lax

        return lax.cond(pred, true_fun, false_fun, *operands)
    elif pred:
        return true_fun(*operands)
    else:
        return false_fun(*operands)


def select(pred, on_true, on_false):
    if array_api_compat.is_jax_namespace(array_api_compat.array_namespace(on_true)):
        from jax import lax

        return lax.select(pred, on_true, on_false)
    else:
        xp = array_api_compat.array_namespace(on_true)
        return xp.where(pred, on_true, on_false)


def scan(f, init, xs=None, length=None):
    if array_api_compat.is_array_api_obj(init):
        xp = array_api_compat.array_namespace(init)
    elif isinstance(init, (float, int, complex)):
        xp = np
    else:
        try:
            xp = array_api_compat.array_namespace(*init)
        except TypeError:
            raise TypeError(
                "init for scan must be an array, scalar or iterable of arrays"
            )

    if array_api_compat.is_jax_namespace(xp):
        from jax import lax

        return lax.scan(f, init, xs, length=length)
    else:
        return _scan(f, init, xs, length=length, xp=xp)


def while_loop(cond_fun, body_fun, init_val):
    if array_api_compat.is_jax_namespace(array_api_compat.array_namespace(init_val)):
        from jax import lax

        return lax.while_loop(cond_fun, body_fun, init_val)
    else:
        return _while_loop(cond_fun, body_fun, init_val)
