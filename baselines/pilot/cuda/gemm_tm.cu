// Tests plan claim: on this GPU the per-step time T(M) of a decode step is memory-bound,
// so T(M) ~= T(1) for small M (M = tokens processed per step). Measures cuBLAS bf16 GEMM
// for every linear layer of Qwen2.5-1.5B and a naive custom weight-streaming kernel.
// Usage: ./gemm_tm [hidden=1536] [inter=8960] [layers=28] [kv_dim=256] [vocab=151936]
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#define CK(x) do { cudaError_t err_ = (x); if (err_ != cudaSuccess) { printf("CUDA error %s at %d\n", cudaGetErrorString(err_), __LINE__); exit(1);} } while(0)
#define CB(x) do { cublasStatus_t err_ = (x); if (err_ != CUBLAS_STATUS_SUCCESS) { printf("cuBLAS error %d at %d\n", (int)err_, __LINE__); exit(1);} } while(0)

static float med(std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size()/2]; }

// Naive weight-streaming kernel: one warp per output row n; each lane strides over K reading
// bf16x2, accumulates for all M tokens (activations X[M][K] bf16 are tiny, L2/L1 resident).
// Y[M][N] = X[M][K] * W[N][K]^T   (W row-major [N][K] like HF Linear weight)
template <int MMAX>
__global__ void gemv_ws(const __nv_bfloat162* __restrict__ W, const __nv_bfloat162* __restrict__ X,
                        float* __restrict__ Y, int N, int K, int M) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= N) return;
    int K2 = K >> 1;
    const __nv_bfloat162* w = W + (size_t)warp * K2;
    float acc[MMAX];
#pragma unroll
    for (int m = 0; m < MMAX; ++m) acc[m] = 0.f;
    for (int k = lane; k < K2; k += 32) {
        float2 wv = __bfloat1622float2(w[k]);
#pragma unroll
        for (int m = 0; m < MMAX; ++m) {
            if (m < M) {
                float2 xv = __bfloat1622float2(X[(size_t)m * K2 + k]);
                acc[m] += wv.x * xv.x + wv.y * xv.y;
            }
        }
    }
#pragma unroll
    for (int m = 0; m < MMAX; ++m) {
        float v = acc[m];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
        if (lane == 0 && m < M) Y[(size_t)m * N + warp] = v;
    }
}

struct Lin { const char* name; int N, K; int count_per_layer; };

int main(int argc, char** argv) {
    int H = argc > 1 ? atoi(argv[1]) : 1536;
    int I = argc > 2 ? atoi(argv[2]) : 8960;
    int L = argc > 3 ? atoi(argv[3]) : 28;
    int KV = argc > 4 ? atoi(argv[4]) : 256;
    int V = argc > 5 ? atoi(argv[5]) : 151936;
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int memclk_khz = 0; CK(cudaDeviceGetAttribute(&memclk_khz, cudaDevAttrMemoryClockRate, 0));
    double peak_bw = 2.0 * memclk_khz * 1e3 * (prop.memoryBusWidth / 8) / 1e9;

    // Qwen2-style layer: q(H->H) k(H->KV) v(H->KV) o(H->H) gate(H->I) up(H->I) down(I->H)
    std::vector<Lin> lins = {
        {"q_proj",  H, H, 1}, {"k_proj", KV, H, 1}, {"v_proj", KV, H, 1}, {"o_proj", H, H, 1},
        {"gate_proj", I, H, 1}, {"up_proj", I, H, 1}, {"down_proj", H, I, 1},
    };
    Lin lm = {"lm_head", V, H, 1};
    double layer_bytes = 0;
    for (auto& l : lins) layer_bytes += 2.0 * l.N * l.K;
    double step_bytes = layer_bytes * L + 2.0 * lm.N * lm.K;
    printf("GPU %s theoretical %.0f GB/s. Linear-weight bytes/step: %.3f GB (layers %.3f GB + lm_head %.3f GB)\n",
           prop.name, peak_bw, step_bytes / 1e9, layer_bytes * L / 1e9, 2.0 * lm.N * lm.K / 1e9);
    printf("Bandwidth floor for one step at theoretical peak: %.2f ms\n\n", step_bytes / peak_bw / 1e9 * 1e3);

    // Allocate a big weight pool so repeated calls stream different weights (defeat the 32MB L2):
    // allocate ALL layer weights (L copies) and rotate through them.
    size_t pool_elems = (size_t)(layer_bytes / 2) * L;
    __nv_bfloat16* pool; CK(cudaMalloc(&pool, pool_elems * 2));
    CK(cudaMemset(pool, 0, pool_elems * 2));
    __nv_bfloat16* lmw; CK(cudaMalloc(&lmw, (size_t)V * H * 2)); CK(cudaMemset(lmw, 0, (size_t)V * H * 2));
    const int MMAX = 512;
    __nv_bfloat16* X; CK(cudaMalloc(&X, (size_t)MMAX * I * 2)); CK(cudaMemset(X, 0, (size_t)MMAX * I * 2));
    __nv_bfloat16* Yb; CK(cudaMalloc(&Yb, (size_t)MMAX * V * 2));
    float* Yf; CK(cudaMalloc(&Yf, (size_t)MMAX * V * 4));

    cublasHandle_t hb; CB(cublasCreate(&hb));
    size_t ws_sz = 64 << 20; void* ws; CK(cudaMalloc(&ws, ws_sz)); CB(cublasSetWorkspace(hb, ws, ws_sz));
    cudaEvent_t s, e; CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e));

    int Ms[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
    printf("%-6s | %-14s %-12s %-10s | %-14s %-10s | %-10s\n", "M", "cuBLAS step(ms)", "GB/s eff", "x T(1)", "naive step(ms)", "GB/s eff", "lm_head(ms)");
    float t1_cublas = 0, t1_naive = 0;
    for (int M : Ms) {
        float alpha = 1.f, beta = 0.f;
        // ---- cuBLAS: run a full "step" = L layers x 7 linears + lm_head, weights rotating through pool
        auto run_cublas = [&]() {
            size_t off = 0;
            for (int l = 0; l < L; ++l) {
                for (auto& li : lins) {
                    // Y^T[N][M] = W[N][K] * X^T[K][M]  => cublas col-major: C(NxM) = A(NxK,row-major W => op T) ...
                    // Use: C = W * X^T with W as K x N col-major (i.e. op=T of row-major [N][K])
                    CB(cublasGemmEx(hb, CUBLAS_OP_T, CUBLAS_OP_N, li.N, M, li.K, &alpha,
                                    pool + off, CUDA_R_16BF, li.K, X, CUDA_R_16BF, li.K, &beta,
                                    Yb, CUDA_R_16BF, li.N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                    off += (size_t)li.N * li.K;
                }
            }
        };
        auto run_lm = [&]() {
            CB(cublasGemmEx(hb, CUBLAS_OP_T, CUBLAS_OP_N, V, M, H, &alpha, lmw, CUDA_R_16BF, H, X, CUDA_R_16BF, H,
                            &beta, Yb, CUDA_R_16BF, V, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        };
        float ms; std::vector<float> t;
        for (int w = 0; w < 2; ++w) run_cublas();
        for (int r = 0; r < 5; ++r) {
            CK(cudaEventRecord(s)); run_cublas(); CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
            CK(cudaEventElapsedTime(&ms, s, e)); t.push_back(ms);
        }
        float tc = med(t);
        t.clear();
        for (int w = 0; w < 2; ++w) run_lm();
        for (int r = 0; r < 5; ++r) {
            CK(cudaEventRecord(s)); run_lm(); CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
            CK(cudaEventElapsedTime(&ms, s, e)); t.push_back(ms);
        }
        float tlm = med(t);

        // ---- naive weight-streaming kernel (M <= 32 only; register-blocked)
        float tn = -1;
        if (M <= 32) {
            auto run_naive = [&]() {
                size_t off = 0;
                for (int l = 0; l < L; ++l) {
                    for (auto& li : lins) {
                        int warps = li.N; int threads = 256; int blocks = (warps * 32 + threads - 1) / threads;
                        const __nv_bfloat162* W2 = (const __nv_bfloat162*)(pool + off);
                        const __nv_bfloat162* X2 = (const __nv_bfloat162*)X;
                        if (M <= 1) gemv_ws<1><<<blocks, threads>>>(W2, X2, Yf, li.N, li.K, M);
                        else if (M <= 4) gemv_ws<4><<<blocks, threads>>>(W2, X2, Yf, li.N, li.K, M);
                        else if (M <= 8) gemv_ws<8><<<blocks, threads>>>(W2, X2, Yf, li.N, li.K, M);
                        else if (M <= 16) gemv_ws<16><<<blocks, threads>>>(W2, X2, Yf, li.N, li.K, M);
                        else gemv_ws<32><<<blocks, threads>>>(W2, X2, Yf, li.N, li.K, M);
                        off += (size_t)li.N * li.K;
                    }
                }
            };
            t.clear();
            for (int w = 0; w < 2; ++w) run_naive();
            CK(cudaGetLastError());
            for (int r = 0; r < 5; ++r) {
                CK(cudaEventRecord(s)); run_naive(); CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
                CK(cudaEventElapsedTime(&ms, s, e)); t.push_back(ms);
            }
            tn = med(t);
        }
        if (M == 1) { t1_cublas = tc; t1_naive = tn; }
        double lb = layer_bytes * L;
        printf("%-6d | %-14.3f %-12.1f %-10.2f | ", M, tc, lb / (tc * 1e-3) / 1e9, tc / t1_cublas);
        if (tn > 0) printf("%-14.3f %-10.1f | ", tn, lb / (tn * 1e-3) / 1e9); else printf("%-14s %-10s | ", "-", "-");
        printf("%-10.3f\n", tlm);
    }
    return 0;
}
