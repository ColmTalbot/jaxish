# jaxish

A drop-in replacement for `jax` operations that are not available in, e.g., `numpy` to enable use of `jax` features while still maintaining compatibility with other libraries.

This library is at a _very_ early stage of development and functionality is limited.
If this is useful to you and you would like to see more features, please open an issue or a pull request.

## Scope

The intended eventual scope is to provide a drop-in replacement for the functions in the `jax` top-level and `jax.lax` namespaces.
For `jax.numpy`, similar functionality is provided via the [Python array API](https://data-apis.org/array-api-compat/).

## Supported Features

| Feature     | NumPy | JAX | MLX | CuPy |
|-------------|:-----:|:---:|:---:|:----:|
| `jit`       | 🚫    | ⚡   | ⚡   | 🚫   |
| `grad`      | ❌    | ⚡   | ⚡   | ❌   |
| `vmap`      | 🐢    | ⚡   | ⚡   | 🐢   |
| `lax.cond`       | 🐢    | ⚡   | 🐢   | 🐢   |
| `lax.scan`       | 🐢    | ⚡   | 🐢   | 🐢   |
| `lax.select`     | 🐢    | ⚡   | 🐢   | 🐢   |
| `lax.while_loop` | 🐢    | ⚡   | 🐢   | 🐢   |

- ⚡ **Native implementation** (uses backend's optimized version)
- 🐢 **Unoptimized fallback** (Python implementation, not optimized)
- 🚫 **No-op** (function does nothing)
- ❌ **Not implemented** (raises an error)
- ⏳ **Not yet implemented** (planned, not available)

## Installation

```console
pip install .[jax]      # For JAX support
pip install .[mlx]      # For MLX support
pip install .[cupy]     # For CuPy support
pip install .           # For NumPy only
```

## Usage Examples

### JIT Compilation
```python
from jaxish import jit
import numpy as np

@jit
def f(x):
    return x * x

print(f(np.array([1.0, 2.0, 3.0])))  # NumPy: [1. 4. 9.]

# With JAX
import jax.numpy as jnp
print(f(jnp.array([1.0, 2.0, 3.0])))  # JAX: [1. 4. 9.]
```

### Differentiation
```python
from jaxish import grad
import jax.numpy as jnp

def f(x):
    return jnp.sum(x ** 2)

g = grad(f)
print(g(jnp.array([1.0, 2.0, 3.0])))  # [2. 4. 6.]
```

### lax
```python
from jaxish import lax
import numpy as np

def f(carry, x):
    return carry + x, carry * x
carry = np.array(1.0)
xs = np.array([2.0, 3.0, 4.0])
final_carry, ys = lax.scan(f, carry, xs)
print(final_carry)  # 10.0
print(ys)           # [2. 6. 24.]

result = lax.cond(True, lambda x: x + 1, lambda x: x - 1, xs)  # array([3., 4., 5.])

pred = np.array([True, False])
on_true = np.array([1, 2])
on_false = np.array([10, 20])
out = lax.select(pred, on_true, on_false)  # array([1, 20])

# while_loop
def cond_fun(val): return val < 5
def body_fun(val): return val + 1
final = lax.while_loop(cond_fun, body_fun, 0)  # 5
```

### vmap
```python
from jaxish import vmap
import numpy as np

@vmap
def f(x):
    return x * 2

arr = np.array([1, 2, 3])
print(f(arr))  # [2 4 6]

# With JAX
import jax.numpy as jnp
@vmap
def f_jax(x):
    return x * 2
print(f_jax(jnp.array([1, 2, 3])))  # [2 4 6]
```

## Testing

To run tests and check coverage:

```fish
pip install .[test,jax,mlx]
pytest
```

## License

MIT
