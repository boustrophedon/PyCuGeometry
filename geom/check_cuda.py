import pycuda
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

import numpy as np

def test_pycuda_works():
  attribs = [
    pycuda.driver.device_attribute.MAX_THREADS_PER_BLOCK,
    pycuda.driver.device_attribute.MAX_BLOCK_DIM_X,
    pycuda.driver.device_attribute.MAX_BLOCK_DIM_Y,
    pycuda.driver.device_attribute.MAX_BLOCK_DIM_Z,
    pycuda.driver.device_attribute.MAX_GRID_DIM_X,
    pycuda.driver.device_attribute.MAX_GRID_DIM_Y,
    pycuda.driver.device_attribute.MAX_GRID_DIM_Z,
    pycuda.driver.device_attribute.TOTAL_CONSTANT_MEMORY,
    pycuda.driver.device_attribute.WARP_SIZE,
    pycuda.driver.device_attribute.MAX_PITCH,
    pycuda.driver.device_attribute.CLOCK_RATE,
    pycuda.driver.device_attribute.TEXTURE_ALIGNMENT,
    pycuda.driver.device_attribute.GPU_OVERLAP,
    pycuda.driver.device_attribute.MULTIPROCESSOR_COUNT,
    pycuda.driver.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK,
    pycuda.driver.device_attribute.MAX_REGISTERS_PER_BLOCK,
    pycuda.driver.device_attribute.KERNEL_EXEC_TIMEOUT,
    pycuda.driver.device_attribute.INTEGRATED,
    pycuda.driver.device_attribute.CAN_MAP_HOST_MEMORY,
    pycuda.driver.device_attribute.COMPUTE_MODE,
  ]

  device = pycuda.autoinit.device

  print()
  for attr in attribs:
    val = device.get_attribute(attr)
    print(attr, val)
    assert val is not None

TIMES2_SRC = \
"""    
__global__ void times2(float *src, float *dst, size_t len) {
  const int i = gridDim.x*blockIdx.x + threadIdx.x;
  if (i < len) {
    dst[i] = 2*src[i];
  }
}
"""


def test_pycuda_times2():
  mod = SourceModule(TIMES2_SRC)

  times2 = mod.get_function("times2")

  src = np.ones((1000, 1), dtype=np.float32)
  dst = np.zeros_like(src)

  times2(
      drv.In(src), drv.Out(dst), np.uintp(len(src)),
      block=(32,1,1), grid=(32,1,1))
  assert (dst == 2*src).all()
