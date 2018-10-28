import os

import numpy as np
from pycuda.compiler import SourceModule

import geom


def load_src(f, *args):
  """ Read a file from the "cuda_src" directory inside this module and load it as a
  cuda program.
  
  `f` is a path to a file relative to the cuda_src directory.
  
  `*args` should be a sequence of strings corresponding to the names of cuda functions.

  Returns a dict of "function_name":cuda_function
  """
  geom_dir = os.path.dirname(geom.__file__)
  cuda_src_dir = os.path.join(geom_dir, "cuda_src", f)

  src_text = None
  with open(cuda_src_dir) as cuda_src_file:
    src_text = cuda_src_file.read()

  mod = SourceModule(src_text)

  funcs = dict()
  for arg in args:
    funcs[arg] = mod.get_function(arg)

  return funcs

def test_load_src():
  import pycuda
  import pycuda.autoinit

  test1 = load_src(".test.cu", "test1")["test1"]
  time = test1(block=(1,1,1), time_kernel = True)
  assert time is not None
  assert time > 0

def test_load_src_multiple():
  import pycuda
  import pycuda.driver as drv
  import pycuda.autoinit

  funcs = load_src(".test.cu", "test1", "test2")
  test1, test2 = funcs["test1"], funcs["test2"]
  test1(block=(1,1,1))

  out = np.zeros((1,), dtype=np.float32)
  test2(drv.Out(out), block=(1,1,1))
  assert out[0] == 1.0
