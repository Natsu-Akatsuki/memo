#include <iostream>

#define N 10

using namespace std;

__global__ void add(int *a, int *b, int *c) {
    int tid = blockIdx.x;
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}

int main(void) {
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    cudaMalloc((void **) &dev_a, N * sizeof(int));
    cudaMalloc((void **) &dev_b, N * sizeof(int));
    cudaMalloc((void **) &dev_c, N * sizeof(int));

    for (int i = 0; i < N; i++) {
        a[i] = -i;
        b[i] = i * 1;
    }

    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyDeviceToDevice);

    add<<<N, 1>>>(dev_a, dev_b, dev_c);

    cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToDevice);

    // 显示结果
    for (int i = 0; i < N; i++) {
        cout << "a[i]+b[i]+c[i]=" << a[i] + b[i] + c[i] << endl;
    }

    // 释放内存（todo：不释放会如何）
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
