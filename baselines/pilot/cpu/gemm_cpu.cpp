// Tests plan claim: on CPU, T(M)/T(1) is much larger than on GPU (CPU is compute-bound
// for M=8..32), so the diffusion break-even point is much higher on CPU.
// bf16 weights [N][K], fp32 activations [M][K]; OpenMP over N, AVX2+FMA inner loop.
// Build: g++ -O3 -march=native -fopenmp gemm_cpu.cpp -o gemm_cpu
#include <immintrin.h>
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>

static inline __m256 load_bf16x8(const uint16_t* p) {
    __m128i h = _mm_loadu_si128((const __m128i*)p);
    __m256i w = _mm256_cvtepu16_epi32(h);
    return _mm256_castsi256_ps(_mm256_slli_epi32(w, 16));
}

// Y[M][N] = X[M][K] * W[N][K]^T. Register-block over MB rows of X.
template <int MB>
static void gemm_block(const uint16_t* W, const float* X, float* Y, int N, int K, int M) {
#pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const uint16_t* w = W + (size_t)n * K;
        for (int m0 = 0; m0 < M; m0 += MB) {
            __m256 acc[MB];
            for (int i = 0; i < MB; ++i) acc[i] = _mm256_setzero_ps();
            for (int k = 0; k < K; k += 8) {
                __m256 wv = load_bf16x8(w + k);
                for (int i = 0; i < MB; ++i) {
                    int m = m0 + i; if (m < M) {
                        __m256 xv = _mm256_loadu_ps(X + (size_t)m * K + k);
                        acc[i] = _mm256_fmadd_ps(wv, xv, acc[i]);
                    }
                }
            }
            for (int i = 0; i < MB; ++i) {
                int m = m0 + i; if (m < M) {
                    float tmp[8]; _mm256_storeu_ps(tmp, acc[i]);
                    Y[(size_t)m * N + n] = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
                }
            }
        }
    }
}

static double med(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size()/2]; }

struct Lin { const char* name; int N, K; };

int main(int argc, char** argv) {
    int H = argc > 1 ? atoi(argv[1]) : 1536, I = argc > 2 ? atoi(argv[2]) : 8960, L = argc > 3 ? atoi(argv[3]) : 28, KV = argc > 4 ? atoi(argv[4]) : 256;
    std::vector<Lin> lins = {{"q",H,H},{"k",KV,H},{"v",KV,H},{"o",H,H},{"gate",I,H},{"up",I,H},{"down",H,I}};
    size_t layer_elems = 0; for (auto& l : lins) layer_elems += (size_t)l.N * l.K;
    size_t pool = layer_elems * L; // all layers: 2.8 GB of bf16, defeats 30 MB L3
    printf("threads=%d, layer weights x%d = %.2f GB bf16\n", omp_get_max_threads(), L, pool * 2.0 / 1e9);
    std::vector<uint16_t> W(pool); for (size_t i = 0; i < pool; i += 4096) W[i] = 0x3f80; // sparse init to touch pages
#pragma omp parallel for
    for (size_t i = 0; i < pool; ++i) W[i] = (uint16_t)(0x3f80 ^ (i & 0xff));
    const int MMAX = 64;
    std::vector<float> X((size_t)MMAX * I, 0.5f), Y((size_t)MMAX * I, 0.f);

    // Pure bandwidth reference: sum the whole pool
    {
        std::vector<double> t;
        for (int r = 0; r < 3; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            uint64_t acc = 0;
#pragma omp parallel for reduction(+:acc)
            for (size_t i = 0; i < pool; i += 32) acc += W[i];
            auto t1 = std::chrono::steady_clock::now();
            t.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            if (acc == 1) printf("x");
        }
        printf("DRAM stream read of %.2f GB: %.1f ms -> %.1f GB/s\n", pool*2.0/1e9, med(t), pool*2.0/(med(t)*1e-3)/1e9);
    }

    int Ms[] = {1, 2, 4, 8, 16, 32, 64};
    double t1 = 0;
    printf("%-4s %-12s %-10s %-8s\n", "M", "step(ms)", "GB/s eff", "x T(1)");
    for (int M : Ms) {
        auto run = [&]() {
            size_t off = 0;
            for (int l = 0; l < L; ++l) for (auto& li : lins) {
                const uint16_t* w = W.data() + off;
                if (M <= 1) gemm_block<1>(w, X.data(), Y.data(), li.N, li.K, M);
                else if (M <= 2) gemm_block<2>(w, X.data(), Y.data(), li.N, li.K, M);
                else if (M <= 4) gemm_block<4>(w, X.data(), Y.data(), li.N, li.K, M);
                else gemm_block<8>(w, X.data(), Y.data(), li.N, li.K, M);
                off += (size_t)li.N * li.K;
            }
        };
        run();
        std::vector<double> t;
        for (int r = 0; r < 3; ++r) {
            auto a = std::chrono::steady_clock::now(); run(); auto b = std::chrono::steady_clock::now();
            t.push_back(std::chrono::duration<double, std::milli>(b - a).count());
        }
        double tm = med(t); if (M == 1) t1 = tm;
        printf("%-4d %-12.1f %-10.1f %-8.2f\n", M, tm, pool * 2.0 / (tm * 1e-3) / 1e9, tm / t1);
    }
    return 0;
}
