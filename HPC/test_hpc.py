from numba import jit, vectorize, cuda
from math import sqrt, erf
import numpy as np
import time

threshold = 0.
SQRT1_2 = 1.0/sqrt(2.)

@cuda.jit(device=True)
def get_cdf_gpu(mu, sigma):
  temp = (threshold - mu)*SQRT1_2 / sigma
  cdf = .5 * (1.+erf(temp))
  return cdf

@cuda.jit
def get_ibv_from_gpu(Mu, Sigma, IBV, EIBV):
  # idx = cuda.grid(1)
  start = cuda.grid(1)
  stride = cuda.gridsize(1)
  for i in range(start, Mu.shape[0], stride):
    IBV[i] = get_cdf_gpu(Mu[i], Sigma[i])*(1-get_cdf_gpu(Mu[i], Sigma[i]))
    cuda.atomic.add(EIBV, 0, IBV[i])

def get_eibv_from_gpu(d_mu, sigma, d_ibv, d_eibv):
  # d_mu = cuda.to_device(mu)
  d_sigma = cuda.to_device(sigma)
  # d_ibv = cuda.device_array_like(d_mu)
  get_ibv_from_gpu[40, 512](d_mu, d_sigma, d_ibv, d_eibv)
  eibv = d_eibv.copy_to_host()
  return eibv


N = 20000
mu = np.linspace(-3, 3, N)
sigma = np.ones_like(mu)

d_mu = cuda.to_device(mu)
d_ibv = cuda.device_array_like(d_mu)
d_eibv = cuda.to_device(np.array([0]).astype(np.float32))
t1 = time.time()
ibv4 = get_eibv_from_gpu(d_mu, sigma, d_ibv, d_eibv)
t2 = time.time()
print("Time consumed: ", t2 - t1)

#%%
from numba import cuda
import numpy as np

@cuda.jit
def add(x, out):
  id = cuda.grid(1)
  out[id] = x[id] +2

N = 20000
x = np.arange(N)
d_x = cuda.to_device(x)
d_out = cuda.device_array_like(d_x)
add[40,512](d_x, d_out)

print("hello ")
#%%
from numba import jit, vectorize, cuda
from math import sqrt, erf
import numpy as np
import time
from scipy.stats import norm
import matplotlib.pyplot as plt


threshold = 0.
SQRT1_2 = 1.0/sqrt(2.)


def get_eibv_from_cpu(mu, sigma):
      p = norm.cdf(threshold, mu, sigma)
      bv = p * (1-p)
      ibv = np.sum(bv)
      return ibv


vectorize(['float32(float32, float32)'], target='gpu')
def get_eibv_from_para(mu, sigma):
  p = norm.cdf(threshold, mu, sigma)
  bv = p*(1-p)
  ibv = np.sum(bv)
  return ibv

N = 20000
mu = np.linspace(-3, 3, N)
sigma = np.ones_like(mu)

n = 1000
dt_cpu = []
dt_gpu = []
for i in range(n):
  print(i)
  t1 = time.time()
  ibv1 = get_eibv_from_cpu(mu, sigma)
  t2 = time.time()
  dt_cpu.append(t2 - t1)


  t1 = time.time()
  ibv2 = get_eibv_from_para(mu.astype(np.float32), sigma.astype(np.float32))
  t2 = time.time()
  dt_gpu.append(t2 - t1)


print("CPU time: ", np.mean(dt_cpu))
print("GPU time: ", np.mean(dt_gpu))
print("GPU speed up: ", np.mean(dt_cpu)/np.mean(dt_gpu))
plt.plot(dt_gpu, label='GPU')
plt.plot(dt_cpu, label='CPU')
plt.legend()
plt.show()

