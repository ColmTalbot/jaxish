from functools import partial, wraps
from inspect import signature

from array_api_compat import array_namespace

__all__ = ["vmap"]


def advance_axis(arr, n):
    """
    Move the first axis forward by n positions.
    E.g., (0, 1, 2, 3, 4) -> (1, 2, 0, 3, 4) for n=2
    """
    if n == 0:
        return arr
    xp = array_namespace(arr)
    ndim = arr.ndim
    if not (0 < n < ndim):
        raise ValueError(f"n must be in the range 1 <= n < arr.ndim, got n={n}")
    axes = list(range(ndim))
    new_axes = axes[1 : n + 1] + [0] + axes[n + 1 :]
    return xp.moveaxis(arr, axes, new_axes)


def _vmap(func, in_axes=0, out_axes=0):
    parameters = signature(func).parameters

    if isinstance(in_axes, int):
        in_axes = tuple(in_axes for _ in parameters)

    @wraps(func)
    def wrapper(*args, **kwargs):
        if len(args) != len(in_axes):
            raise ValueError(
                f"Expected to map over {len(parameters)} positional arguments,"
                f" but got {len(args)} positional arguments"
            )

        args = tuple(arg.swapaxes(axis, 0) for axis, arg in zip(in_axes, args))
        xp = array_namespace(args[0])

        map_length = {len(arg) for arg in args}
        if len(map_length) != 1:
            raise ValueError(
                "All arguments must have the same length along the mapped axis"
            )
        map_length = map_length.pop()

        output = []
        for ii in range(map_length):
            output.append(func(*[arg[ii] for arg in args], **kwargs))

        if isinstance(output[0], tuple):
            if isinstance(out_axes, int):
                axes = tuple(out_axes for _ in output[0])
            else:
                axes = out_axes
            output = tuple(
                advance_axis(xp.stack([out[i] for out in output]), axes[i])
                for i in range(len(output[0]))
            )
        else:
            output = advance_axis(xp.stack(output), out_axes)

        return output

    return wrapper


def vmap(func, in_axes=0, out_axes=0, **vmap_kwargs):
    """
    A decorator to vectorize a function. Uses jax.vmap or mlx.core.vmap if available,
    otherwise falls back to a simple Python implementation.
    """

    @wraps(func)
    def wrapped(*args, **kwargs):
        xp = array_namespace(*args)

        if "jax" in xp.__name__:
            from jax import vmap as vmap_func

            vmap_func = partial(vmap_func, **vmap_kwargs)
        elif "mlx" in xp.__name__:
            from mlx.core import vmap as vmap_func
        else:
            vmap_func = _vmap

        return vmap_func(func, in_axes=in_axes, out_axes=out_axes)(*args, **kwargs)

    return wrapped
