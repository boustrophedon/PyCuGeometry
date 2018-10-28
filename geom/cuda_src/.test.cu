__global__ void test1() {
  return;
}

__global__ void test2(float *x) {
  x[0] = 1;
  return;
}
