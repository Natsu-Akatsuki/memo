#include <iostream>

const int N = 33 * 1024;
const int threadsPerBlock = 256;

__global__ void dot(float *a, float *b, float *c) {

    // 声明一个共享缓存区
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    float temp = 0;

    while (tid < N) {
        temp += a[tid] * b[tid];
        tid += blockDim.x + gridDim.x;
    }

    cache[cacheIndex] = temp;
}

const int blocksPerGrid = imin(32,(N+threadsPerBlock-1/threadsPerBlock));


int main() {
    float
}
