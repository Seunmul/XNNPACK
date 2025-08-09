#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>

#include "bench/utils.h"
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/config-types.h"
#include "src/xnnpack/config.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/pack-lh.h"
#include "src/xnnpack/pack.h"
#include "src/xnnpack/packq.h"
#include "src/xnnpack/packw.h"

#include "gemm-profiler-interface.h"
#include <type_traits> // for std::is_integral_v, std::is_floating_point_v

static inline uint64_t now_ns() {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
    return 0;
  }
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}


/******************************************************************************
 * 템플릿 기반의 핵심 프로파일러 함수 (설계도)
 ******************************************************************************/
template <
    typename T,           // 데이터 타입 (예: float, int8_t)
    typename TParam,      // 파라미터 구조체 타입 (예: xnn_f32_minmax_params)
    typename UKernelFn,   // ukernel 함수 포인터 타입
    typename InitParamsFn,// init 함수 포인터 타입
    typename PackFn       // pack 함수 포인터 타입
>
double xnn_profile_gemm_bench(
    UKernelFn gemm,
    InitParamsFn init_params,
    PackFn pack,
    size_t mr, size_t nr, size_t kr, size_t sr,
    size_t mc, size_t nc, size_t kc,
    size_t warmup_iters, size_t measure_iters)
{
    if (!gemm || !init_params || !pack || mc == 0 || nc == 0 || kc == 0) {
        return 0.0;
    }


    // --- 기존 GEMMBenchmark 로직 시작 ---
    const size_t nc_stride = benchmark::utils::RoundUp(nc, nr);
    const size_t kc_stride = benchmark::utils::RoundUp(kc, kr * sr);

    std::random_device rd;
    auto rng = std::mt19937(rd());
    
    // 데이터 타입에 따라 다른 랜덤 값 생성기 사용 (C++11/14 호환)
    auto dist_gen = [&]() {
        if (std::is_floating_point<T>::value) {
            return std::uniform_real_distribution<T>()(rng);
        } else if (std::is_integral<T>::value) {
            return static_cast<T>(std::uniform_int_distribution<int32_t>(
                std::numeric_limits<T>::min(), std::numeric_limits<T>::max())(rng));
        }
    };

    // 입력, 가중치, 바이어스 버퍼 생성 및 초기화
    xnnpack::Buffer<T> a(mc * kc + XNN_EXTRA_BYTES);
    std::generate(a.begin(), a.end(), dist_gen);
    xnnpack::Buffer<T> k(nc * kc);
    std::generate(k.begin(), k.end(), dist_gen);
    xnnpack::Buffer<T> b(nc);
    std::generate(b.begin(), b.end(), dist_gen);

    // 캐시 효과를 줄이기 위한 버퍼 순환 로직
    const size_t w_elements = nc_stride * kc_stride + nc_stride;
    const size_t c_elements = mc * nc;
    const size_t num_buffers = 1 + benchmark::utils::DivideRoundUp<size_t>(
        benchmark::utils::GetMaxCacheSize(),
        sizeof(T) * (w_elements + c_elements));

    xnnpack::Buffer<T, XNN_ALLOCATION_ALIGNMENT> w(w_elements * num_buffers);
    pack(/*groups=*/1, nc, kc, nr, kr, sr,
         k.data(), b.data(), /*scale=*/nullptr,
         w.data(), /*extra_bytes=*/0, /*params=*/nullptr);

    xnnpack::Buffer<T> c(c_elements * num_buffers);
    std::fill(c.begin(), c.end(), 0xA5); // 쓰레기 값으로 채움

    // 파라미터 초기화 (표준 라이브러리 사용)
    TParam params;
    init_params(&params, 
                std::numeric_limits<T>::lowest(),
                std::numeric_limits<T>::max());

    // 워밍업 실행
    size_t buffer_index = 0;
    for (size_t it = 0; it < warmup_iters; it++) {
        for (uint32_t m = 0; m < mc; m += mr) {
            const uint32_t mb = std::min<uint32_t>(mc - m, (uint32_t)mr);
            gemm(mb, (uint32_t)nc, (uint32_t)(kc * sizeof(T)),
                 a.data() + m * kc, (uint32_t)(kc * sizeof(T)),
                 w.data() + buffer_index * w_elements,
                 c.data() + (buffer_index * mc + m) * nc, (uint32_t)(nc * sizeof(T)),
                 (uint32_t)(nr * sizeof(T)), &params);
        }
        buffer_index = (buffer_index + 1) % num_buffers;
    }
    
    // 성능 측정 실행
    const uint64_t t0 = now_ns();
    for (size_t it = 0; it < measure_iters; it++) {
        for (uint32_t m = 0; m < mc; m += mr) {
            const uint32_t mb = std::min<uint32_t>(mc - m, (uint32_t)mr);
            gemm(mb, (uint32_t)nc, (uint32_t)(kc * sizeof(T)),
                 a.data() + m * kc, (uint32_t)(kc * sizeof(T)),
                 w.data() + buffer_index * w_elements,
                 c.data() + (buffer_index * mc + m) * nc, (uint32_t)(nc * sizeof(T)),
                 (uint32_t)(nr * sizeof(T)), &params);
        }
        buffer_index = (buffer_index + 1) % num_buffers;
    }
    const uint64_t t1 = now_ns();
    
    // GFLOPS 계산
    const double seconds = double(t1 - t0) / 1e9;
    const double ops = double(measure_iters) * 2.0 * double(mc) * double(nc) * double(kc);
    const double gflops = ops / seconds / 1e9;

    return std::isfinite(gflops) ? gflops : 0.0;
}




// /******************************************************************************
//  * C 코드에서 호출할 수 있는 C-호환 래퍼 함수들 (제품)
//  ******************************************************************************/
extern "C" {

// --- f32용 프로파일러 ---
double xnn_profile_f32_gemm_minmax(
    xnn_f32_gemm_minmax_ukernel_fn gemm,
    xnn_init_f32_minmax_params_fn init_params,
    xnn_pack_f32_gemm_fn pack,
    size_t mr, size_t nr, size_t kr, size_t sr,
    size_t mc, size_t nc, size_t kc,
    size_t warmup_iters, size_t measure_iters)
{
    xnn_log_info(
        "[gemm_profiler_interface] Profiling f32 GEMM with mr=%zu, nr=%zu, kr=%zu, sr=%zu, "
        "mc=%zu, nc=%zu, kc=%zu, warmup_iters=%zu, measure_iters=%zu",
        mr, nr, kr, sr, mc, nc, kc, warmup_iters, measure_iters);
    // f32 타입을 지정해 템플릿 함수를 호출
    return xnn_profile_gemm_bench<float, xnn_f32_minmax_params>(
        gemm, init_params, pack, mr, nr, kr, sr, mc, nc, kc, warmup_iters, measure_iters);
}


} // extern "C"
