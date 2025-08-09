// gemm_profiler_interface.h

#ifndef GEMM_PROFILER_INTERFACE_H_
#define GEMM_PROFILER_INTERFACE_H_

#include <stdint.h> // C 표준 헤더 사용
#include <stddef.h> // C 표준 헤더 사용



#include "src/xnnpack/microfnptr.h" // XNNPACK의 마이크로커널 함수 포인터 정의
#include "src/xnnpack/pack.h" // XNNPACK의 패킹 함수 정의

// C 코드에 노출할 함수 목록
#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// --- f32용 프로파일러 ---
double xnn_profile_f32_gemm_minmax(
    xnn_f32_gemm_minmax_ukernel_fn gemm,
    xnn_init_f32_minmax_params_fn init_params,
    xnn_pack_f32_gemm_fn pack,
    size_t mr, size_t nr, size_t kr, size_t sr,
    size_t mc, size_t nc, size_t kc,
    size_t warmup_iters, size_t measure_iters);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // GEMM_PROFILER_INTERFACE_H_
