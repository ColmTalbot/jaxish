# jaxish

A drop-in replacement for `jax` operations that are not available in, e.g., `numpy`.

## Supported Features

| Feature | NumPy | JAX | MLX | CuPy |
|---------|-------|-----|-----|------|
| `jit`   | No-op | ✔️   | ✔️   | No-op |
| `grad`  | ❌    | ✔️   | ✔️   | ❌   |
| `scan`  | ✔️    | ✔️   | ✔️   | ✔️   |

- **jit**: Just-in-time compilation using `jax.jit` or `mlx.core.compile` if available, otherwise a no-op.
- **grad**: Differentiation using `jax.grad` or `mlx.core.grad` if available. Raises `NotImplementedError` for other backends.
- **scan**: Functional scan operation, implemented for all backends. This is not
optimized for anything other than `jax`, but it is a drop-in replacement.

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

### Scan
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
```

## Testing

To run tests and check coverage:

```fish
pip install .[test,jax,mlx]
pytest
```

## License

MIT
