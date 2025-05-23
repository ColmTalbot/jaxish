import array_api_compat
import numpy as np


def _scan(f, init, xs, length=None, *, xp):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, xp.stack(ys)


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
