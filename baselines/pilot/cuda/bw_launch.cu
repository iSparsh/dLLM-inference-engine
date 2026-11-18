// Tests plan claims: (a) ~256 GB/s achievable bandwidth, (b) ~3-5 us per kernel launch,
// (c) CUDA Graphs remove most launch overhead.
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>
#include <algorithm>

#define CK(x) do { cudaError_t err_ = (x); if (err_ != cudaSuccess) { printf("CUDA error %s at %d\n", cudaGetErrorString(err_), __LINE__); exit(1);} } while(0)

__global__ void read_kernel(const int4* __restrict__ p, size_t n, int* out) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    int4 acc = make_int4(0,0,0,0);
    for (; i < n; i += stride) { int4 v = p[i]; acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w; }
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x7fffffff) *out = 1; // never true; prevents DCE
}
__global__ void copy_kernel(const int4* __restrict__ a, int4* __restrict__ b, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) b[i] = a[i];
}
__global__ void empty_kernel(int* p) { if (threadIdx.x == 12345) *p = 0; }

static float med(std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size()/2]; }

int main() {
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int memclk_khz = 0; CK(cudaDeviceGetAttribute(&memclk_khz, cudaDevAttrMemoryClockRate, 0));
    printf("GPU: %s, SMs=%d, memclk=%d kHz, bus=%d bit, L2=%d MB -> theoretical %.1f GB/s\n", prop.name,
           prop.multiProcessorCount, memclk_khz, prop.memoryBusWidth, prop.l2CacheSize >> 20,
           2.0 * memclk_khz * 1e3 * (prop.memoryBusWidth / 8) / 1e9);

    const size_t bytes = 1ull << 30; // 1 GiB (model weights are 3.1 GB, far beyond L2=32MB)
    int4 *a, *b; int* flag;
    CK(cudaMalloc(&a, bytes)); CK(cudaMalloc(&b, bytes)); CK(cudaMalloc(&flag, 4));
    CK(cudaMemset(a, 1, bytes));
    size_t n = bytes / sizeof(int4);
    cudaEvent_t s, e; CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e));
    float ms;

    // --- read-only streaming bandwidth (what a weight-streaming GEMV does) ---
    int grid = prop.multiProcessorCount * 8;
    for (int w = 0; w < 3; ++w) read_kernel<<<grid, 256>>>(a, n, flag);
    std::vector<float> t;
    for (int r = 0; r < 10; ++r) {
        CK(cudaEventRecord(s)); read_kernel<<<grid, 256>>>(a, n, flag); CK(cudaEventRecord(e));
        CK(cudaEventSynchronize(e)); CK(cudaEventElapsedTime(&ms, s, e)); t.push_back(ms);
    }
    printf("read-only stream 1GiB: median %.3f ms -> %.1f GB/s\n", med(t), bytes / (med(t) * 1e-3) / 1e9);

    // --- copy bandwidth (read+write) ---
    t.clear();
    for (int r = 0; r < 10; ++r) {
        CK(cudaEventRecord(s)); copy_kernel<<<grid, 256>>>(a, b, n); CK(cudaEventRecord(e));
        CK(cudaEventSynchronize(e)); CK(cudaEventElapsedTime(&ms, s, e)); t.push_back(ms);
    }
    printf("copy 1GiB (r+w): median %.3f ms -> %.1f GB/s total traffic\n", med(t), 2.0 * bytes / (med(t) * 1e-3) / 1e9);
    t.clear();
    for (int r = 0; r < 10; ++r) {
        CK(cudaEventRecord(s)); CK(cudaMemcpyAsync(b, a, bytes, cudaMemcpyDeviceToDevice)); CK(cudaEventRecord(e));
        CK(cudaEventSynchronize(e)); CK(cudaEventElapsedTime(&ms, s, e)); t.push_back(ms);
    }
    printf("cudaMemcpy D2D 1GiB: median %.3f ms -> %.1f GB/s total traffic\n", med(t), 2.0 * bytes / (med(t) * 1e-3) / 1e9);

    // --- launch overhead: N back-to-back empty kernels ---
    const int N = 336; // plan's estimate of launches/step for Qwen3-0.6B
    for (int w = 0; w < 3; ++w) for (int i = 0; i < N; ++i) empty_kernel<<<1, 32>>>(flag);
    CK(cudaDeviceSynchronize());
    t.clear();
    for (int r = 0; r < 10; ++r) {
        CK(cudaEventRecord(s)); for (int i = 0; i < N; ++i) empty_kernel<<<1, 32>>>(flag); CK(cudaEventRecord(e));
        CK(cudaEventSynchronize(e)); CK(cudaEventElapsedTime(&ms, s, e)); t.push_back(ms);
    }
    printf("%d empty launches (stream): median %.3f ms -> %.2f us/launch\n", N, med(t), med(t) * 1e3 / N);

    // same with CUDA Graph
    cudaStream_t st; CK(cudaStreamCreate(&st));
    cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < N; ++i) empty_kernel<<<1, 32, 0, st>>>(flag);
    CK(cudaStreamEndCapture(st, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    for (int w = 0; w < 3; ++w) CK(cudaGraphLaunch(ge, st));
    CK(cudaStreamSynchronize(st));
    t.clear();
    for (int r = 0; r < 10; ++r) {
        CK(cudaEventRecord(s, st)); CK(cudaGraphLaunch(ge, st)); CK(cudaEventRecord(e, st));
        CK(cudaEventSynchronize(e)); CK(cudaEventElapsedTime(&ms, s, e)); t.push_back(ms);
    }
    printf("%d empty kernels in CUDA Graph: median %.3f ms -> %.2f us/kernel\n", N, med(t), med(t) * 1e3 / N);
    return 0;
}
