#ifndef FIREFLY_POS_H_
#define FIREFLY_POS_H_
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "firefly.h"
#include "../run2DCUDA/defines.h"

__global__ void positions_init_GPU(float *x, curandState *state, int *iteration_number);
__global__ void firefly_random_GPU(float *x, float *alpha0, float *alpha, curandState *state);
__global__ void firefly_random_norm_GPU(float *alpha0, float *alpha, int *iteration_number, int iterations);
__global__ void pos_hist_GPU(float *x, float *pos, float *fitness);
__global__ void ave_pos_hist_GPU(float *x, float *ave_pos, float *fitness);
__global__ void inter_pos_hist_GPU(float *x, float *inter_pos, float *fitness);
__global__ void ave_inter_pos_hist_GPU(float *x, float *ave_inter_pos, float *fitness);

#endif
