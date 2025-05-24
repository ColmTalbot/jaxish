from functools import partial

import numpy as np
import pytest
from jaxish.vmap import vmap


def test_vmap(xp):
    @vmap
    def f(x):
        return x * 2

    arr = xp.array([1, 2, 3])
    out = f(arr)
    assert np.allclose(out, arr * 2)


def test_vmap_tuple_output(xp):
    @vmap
    def f(x):
        return x, x + 1

    arr = xp.array([1, 2, 3])
    out1, out2 = f(arr)
    assert np.allclose(out1, arr)
    assert np.allclose(out2, arr + 1)


def test_vmap_multiple_args(xp):
    @vmap
    def f(x, y):
        return x + y

    arr1 = xp.array([1, 2, 3])
    arr2 = xp.array([4, 5, 6])
    out = f(arr1, arr2)
    assert np.allclose(out, arr1 + arr2)


def test_vmap_advanced_indexing(xp):
    @vmap
    def f(x):
        return x[0]

    arr = xp.array([[1, 2], [3, 4], [5, 6]])
    out = f(arr)
    assert np.allclose(out, arr[:, 0])


def test_vmap_specify_single_in_out_axis(xp):
    @partial(vmap, in_axes=1, out_axes=1)
    def f(x, y):
        return x * 2 + y

    arr = xp.array([[[1, 2], [3, 4], [5, 6]]])
    arr2 = xp.array([[1, 2, 3]])
    print(arr.shape, arr2.shape)
    out = f(arr, arr2)
    assert np.allclose(out, arr * 2 + arr2[:, :, None])


def test_vmap_specify_single_in_axis(xp):
    @partial(vmap, in_axes=1)
    def f(x, y):
        return x * 2 + y

    arr = xp.array([[[1, 2], [3, 4], [5, 6]]])
    arr2 = xp.array([[1, 2, 3]])
    print(arr.shape, arr2.shape)
    out = f(arr, arr2)
    assert np.allclose(out.swapaxes(0, 1), arr * 2 + arr2[:, :, None])


def test_vmap_specify_single_out_axis(xp):
    @partial(vmap, out_axes=1)
    def f(x, y):
        return x * 2 + y

    arr = xp.array([[1, 2], [3, 4], [5, 6]])
    arr2 = xp.array([1, 2, 3])
    print(arr.shape, arr2.shape)
    out = f(arr, arr2)
    print(out.shape)
    assert np.allclose(out.swapaxes(0, 1), arr * 2 + arr2[:, None])


def test_vmap_out_axis_out_of_range(xp):
    @partial(vmap, out_axes=5)
    def f(x, y):
        return x * 2 + y

    arr = xp.array([[1, 2], [3, 4], [5, 6]])
    arr2 = xp.array([1, 2, 3])
    with pytest.raises(ValueError):
        f(arr, arr2)


def test_not_args_number_mismatch_raises(xp):
    def f(x, y, z):
        return x * 2 + y + z

    args = xp.array([1, 2]), xp.array([3, 4]), xp.array([5, 6])

    with pytest.raises(ValueError):
        vmap(f, in_axes=(0, 1))(*args)
    with pytest.raises(ValueError):
        vmap(f, in_axes=(0, 1, 2, 3))(*args)


def test_vmap_mismatched_axes_raises(xp):
    @vmap
    def f(x, y):
        return x * 2 + y

    arr = xp.array([1, 2, 3])
    arr2 = xp.array([1, 3])

    with pytest.raises(ValueError):
        f(arr, arr2)


def test_vmap_specify_multiple_in_axis(xp):
    if xp.__name__ == "mlx.core":
        pytest.skip(reason="Not supported in mlx")

    @partial(vmap, in_axes=(1, 0))
    def f(x, y):
        return x * 2 + y

    arr = xp.array([[1, 2, 3], [4, 5, 6]])
    arr2 = xp.array([[1], [2], [3]])
    out = f(arr, arr2)
    assert np.allclose(out, arr.T * 2 + arr2)


def test_vmap_specify_multiple_out_axis(xp):
    if xp.__name__ == "mlx.core":
        pytest.skip(reason="Not supported in mlx")

    @partial(vmap, in_axes=(1, 1), out_axes=(1, 0))
    def f(x, y):
        return x * 2 + y, x * 3 + y

    arr = xp.array([[1, 2, 3], [4, 5, 6]])
    arr2 = xp.array([[1, 2, 3]])
    out, out2 = f(arr, arr2)
    assert np.allclose(out, arr * 2 + arr2)
    assert np.allclose(out2.swapaxes(0, 1), arr * 3 + arr2)
