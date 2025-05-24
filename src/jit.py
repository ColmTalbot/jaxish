import functools

import numpy as np
from array_api_compat import array_namespace, is_array_api_obj

__all__ = ["jit"]


def jit(func, *jit_args, **jit_kwargs):
    """
    A decorator to jit the function if using jax or mlx and nothing otherwise.

    This also allows arbitrary arguments to be passed through,
    e.g., to specify static arguments.
    """
    default_xp = jit_kwargs.pop("xp", np)

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        xp = kwargs.pop("xp", default_xp)
        if len(args) > 0 and is_array_api_obj(args[0]):
            xp = array_namespace(args[0])

        if "jax" in xp.__name__:
            from jax import jit

            return jit(func, *jit_args, **jit_kwargs)(*args, **kwargs)
        elif "mlx" in xp.__name__:
            from mlx.core import compile

            return compile(func)(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    return wrapped
