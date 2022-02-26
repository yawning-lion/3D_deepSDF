import time
import numpy as np
import jax
from jax import numpy as jnp
from jax.config import config

config.update("jax_enable_x64", True)
np.random.seed(0)

signal = np.random.randn(1024*1024)
signal_jax = jnp.array(signal)

jfft = jax.jit(jnp.fft.fft)

X_np = np.fft.fft(signal)
X_jax = jfft(signal_jax)

print(np.mean(np.abs(X_np)))
print('max:\t', jnp.max(jnp.abs(X_np - X_jax)))
print('mean:\t', jnp.mean(jnp.abs(X_np - X_jax)))
print('min:\t', jnp.min(jnp.abs(X_np - X_jax)))


R = 100
ts = time.time()
for i in range(R):
    _ = np.fft.fft(signal)
time_np = (time.time()-ts)/R * 1000
print('numpy fft execution time [ms]:\t', time_np)

# Compile
_ = jfft(signal_jax).block_until_ready()

ts = time.time()
for i in range(R):
    _ = jfft(signal_jax).block_until_ready()
time_jnp = (time.time()-ts)/R * 1000
print('jax fft execution time [ms]:\t', time_jnp)

print('the speedup due to jax-jit:\t', time_np/time_jnp)