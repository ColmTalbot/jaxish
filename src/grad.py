import functools

import numpy as np
from array_api_compat import array_namespace, is_array_api_obj

__all__ = ["grad"]


def grad(func, *grad_args, **grad_kwargs):
    """
    A decorator to compute the gradient of the function if using jax or mlx.
    Raises NotImplementedError for other backends.
    """

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        xp = kwargs.pop("xp", np)
        if len(args) > 0 and is_array_api_obj(args[0]):
            xp = array_namespace(args[0])

        if "jax" in xp.__name__:
            from jax import grad as jax_grad

            return jax_grad(func, *grad_args, **grad_kwargs)(*args, **kwargs)
        elif "mlx" in xp.__name__:
            from mlx.core import grad as mlx_grad

            return mlx_grad(func, *grad_args, **grad_kwargs)(*args, **kwargs)
        else:
            raise NotImplementedError(
                "grad is only implemented for jax and mlx backends."
            )

    return wrapped
