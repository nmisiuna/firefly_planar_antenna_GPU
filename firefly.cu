#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>

extern "C"{
#include "firefly.h"
#include "printing.h"
#include "../../../nicklib/nicklib.h"
#include "../run2DCUDA/defines.h"
}
#include "../../../nicklib/nicklib_GPU.h"
#include "firefly_pos.h"

static const int iterations = 100;

static float x[FIREFLYNUM * ELEMENTNUM * 2];
static float fitness[FIREFLYNUM];
static float ave_fitness[iterations];
static float beta[FIREFLYNUM];  //Attractiveness
static float r[FIREFLYNUM];  //Distance
static float delta[PHI_RESOL * THETA_RESOL];
static float mean[PHI_RESOL * THETA_RESOL];
static float tempmean[PHI_RESOL * THETA_RESOL];
static float var[PHI_RESOL * THETA_RESOL];
static float M2[PHI_RESOL * THETA_RESOL];
static float FFdes[PHI_RESOL * THETA_RESOL];
static float FF[FIREFLYNUM * PHI_RESOL * THETA_RESOL];
static float cos_phi_sweep[PHI_RESOL];
static float sin_phi_sweep[PHI_RESOL];
static float sin_theta_sweep[THETA_RESOL];
static float pos[POS_RESOL * POS_RESOL * ELEMENTNUM];
static float ave_pos[AVE_POS_RESOL * AVE_POS_RESOL];
static float inter_pos[INTER_POS_RESOL * ELEMENTNUM];
static float ave_inter_pos[INTER_POS_RESOL];

FILE *filepos, *fileavepos, *fileinterpos, *fileaveinterpos, *filed;
FILE *filedes;
FILE *filemean;
FILE *filevar;
FILE *filesolncount;
FILE *filex;
FILE *fileconvergence;

//GPU FUNCTIONS
__global__ void inits_GPU(curandState *state, unsigned long *seed);
__global__ void firefly_init_GPU(float *FFdes, float *cos_phi_sweep, float *sin_phi_sweep, float *sin_theta_sweep, int *iteration_number);
__global__ void firefly_radpat2D_GPU(float *x, float *FF, float *cos_phi_sweep, float *sin_phi_sweep, float *sin_theta_sweep, float *khat);
__global__ void firefly_fitness_GPU(float *FF, float *FFdes, float *fitness);
__global__ void firefly_fitness_norm_GPU(float *fitness);
__global__ void firefly_r_o_GPU(float *x, float *r_o);
__global__ void firefly_sort_x_GPU(float *x, float *r_o);
__global__ void firefly_x_prev_GPU(float *x, float *x_prev);
__global__ void firefly_beta_GPU(float *x, float *beta, float *r);
__global__ void firefly_beta_norm_GPU(float *beta, float *r, float *gamma);
__global__ void firefly_converge_GPU(float *x, float *x_prev, float *beta, float *fitness);
__global__ void moments_counter_GPU(float *fitness, int *mean_counter);
__global__ void moments_GPU(float *FF, float *fitness, float *delta, float *mean, float *M2, int *mean_counter);
__global__ void moments_total_GPU(float *mean, float *mean_total, float *M2, float *M2_total, int *mean_counter, int *mean_counter_total);
__global__ void moments_total_norm_GPU(float *mean_total, float *M2_total, int *mean_counter_total);

//For debugging
//Checks for error and also synchronizes after every gpu function call
inline void check_cuda_errors(const char *filename, const int line_number){
#ifdef DEBUG
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess){
    printf("CUDA error at %s:%i: %s\n", filename, line_number, cudaGetErrorString(error));
    exit(-1);
  }
#endif
}

//Make sure file opened successfully
inline void check_file(FILE **filename){
  if(filename == NULL){
    printf("Can't open %s.txt", filename);
    exit(1);
  }
}

//CPU FUNCTIONS
extern "C" void radpat2D(float FF[], float d[], float khat);
extern "C" float fcost2D(float FF[], float FFdes[]);

extern "C" float firefly(){

  //CPU INITIALIZATIONS
  ///////////////////////////////
  int i, j, k, l, m, n;
  int ensemble_number;
  float test, junk, rand_temp, temp = 0;
  int element, element2;
  float khat = (2.0 * pi / lambda);

  float gamma = Lx * Ly * 2 * ELEMENTNUM * ELEMENTNUM;
  float alpha0 = alpha_w;
  float alpha = alpha0;
  int mean_counter = 0;

  //Seed is fixed if DEBUG flag present
  time_t seed;
  time(&seed);
  #ifdef DEBUG
    srand(1337);
  #else
    srand(time(NULL));
  #endif

  ///////////////////////////////
  //END CPU INITIALIZATION

  //GPU INITIALIZATION
  ///////////////////////////////
  float *d_x, *d_x_prev, *d_r_o, *d_FF, *d_FFdes;
  float *d_cos_phi_sweep, *d_sin_phi_sweep, *d_sin_theta_sweep;
  float *d_beta;
  float *d_fitness, *d_r;
  float *d_gamma, *d_alpha0, *d_alpha;
  int *d_iteration_number;
  curandState *d_state;
  unsigned long *d_seed;
  float *d_khat;
  int *d_mean_counter, *d_mean_counter_total;
  float *d_delta, *d_mean, *d_M2;
  float *d_mean_total, *d_M2_total;
  float *d_pos, *d_ave_pos, *d_inter_pos, *d_ave_inter_pos;

  cudaSetDevice(0);

  float t;
  cudaEvent_t start, stop;

  //Start timing
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  //Allocate GPU arrays
  cudaMalloc((void **) &d_state, N_THREADS * N_BLOCKS * sizeof(curandState));
  cudaMalloc((float **) &d_x, FIREFLYNUM * ELEMENTNUM * 2 * sizeof(float));
  cudaMalloc((float **) &d_x_prev, FIREFLYNUM * ELEMENTNUM * 2 * sizeof(float));
  cudaMalloc((float **) &d_r_o, FIREFLYNUM * ELEMENTNUM * sizeof(float));
  cudaMalloc((float **) &d_FF, FIREFLYNUM * PHI_RESOL * THETA_RESOL * sizeof(float));
  cudaMalloc((float **) &d_FFdes, PHI_RESOL * THETA_RESOL * sizeof(float));
  cudaMalloc((float **) &d_cos_phi_sweep, PHI_RESOL * sizeof(float));
  cudaMalloc((float **) &d_sin_phi_sweep, PHI_RESOL * sizeof(float));
  cudaMalloc((float **) &d_sin_theta_sweep, THETA_RESOL * sizeof(float));
  cudaMalloc((float **) &d_beta, FIREFLYNUM * FIREFLYNUM * sizeof(float));
  cudaMalloc((float **) &d_fitness, FIREFLYNUM * sizeof(float));
  cudaMalloc((float **) &d_r, FIREFLYNUM * FIREFLYNUM * sizeof(float));
  cudaMalloc((float **) &d_gamma, sizeof(float));
  cudaMalloc((float **) &d_alpha0, sizeof(float));
  cudaMalloc((float **) &d_alpha, sizeof(float));
  cudaMalloc((int **) &d_iteration_number, sizeof(int));
  cudaMalloc((unsigned long **) &d_seed, sizeof(unsigned long));
  cudaMalloc((float **) &d_khat, sizeof(float));
  cudaMalloc((int **) &d_mean_counter, FIREFLYNUM * sizeof(int));
  cudaMalloc((int **) &d_mean_counter_total, sizeof(int));
  cudaMalloc((float **) &d_delta, FIREFLYNUM * PHI_RESOL * THETA_RESOL * sizeof(float));
  cudaMalloc((float **) &d_mean, FIREFLYNUM * PHI_RESOL * THETA_RESOL * sizeof(float));
  cudaMalloc((float **) &d_mean_total, PHI_RESOL * THETA_RESOL * sizeof(float));
  cudaMalloc((float **) &d_M2, FIREFLYNUM * PHI_RESOL * THETA_RESOL * sizeof(float));
  cudaMalloc((float **) &d_M2_total, PHI_RESOL * THETA_RESOL * sizeof(float));
  cudaMalloc((float **) &d_pos, POS_RESOL * POS_RESOL * ELEMENTNUM * sizeof(float));
  cudaMalloc((float **) &d_ave_pos, AVE_POS_RESOL * AVE_POS_RESOL * sizeof(float));
  cudaMalloc((float **) &d_inter_pos, INTER_POS_RESOL * ELEMENTNUM * sizeof(float));
  cudaMalloc((float **) &d_ave_inter_pos, AVE_INTER_POS_RESOL * sizeof(float));

  //Move stuff over
  cudaMemcpy(d_gamma, &gamma, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_alpha0, &alpha0, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_alpha, &alpha, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_seed, &seed, sizeof(unsigned long), cudaMemcpyHostToDevice);
  cudaMemcpy(d_khat, &khat, sizeof(float), cudaMemcpyHostToDevice);

  //Initialize statistics
  cudaMemset(d_mean_counter, 0, FIREFLYNUM * sizeof(int));
  cudaMemset(d_mean_counter_total, 0, sizeof(int));
  cudaMemset(d_delta, 0, FIREFLYNUM * PHI_RESOL * THETA_RESOL * sizeof(float));
  cudaMemset(d_mean, 0, FIREFLYNUM * PHI_RESOL * THETA_RESOL * sizeof(float));
  cudaMemset(d_mean_total, 0, PHI_RESOL * THETA_RESOL * sizeof(float));
  cudaMemset(d_M2, 0, FIREFLYNUM * PHI_RESOL * THETA_RESOL * sizeof(float));
  cudaMemset(d_M2_total, 0, PHI_RESOL * THETA_RESOL * sizeof(float));
  cudaMemset(d_pos, 0, POS_RESOL * POS_RESOL * ELEMENTNUM * sizeof(float));
  cudaMemset(d_ave_pos, 0, AVE_POS_RESOL * AVE_POS_RESOL * sizeof(float));
  cudaMemset(d_inter_pos, 0, INTER_POS_RESOL * ELEMENTNUM * sizeof(float));
  cudaMemset(d_ave_inter_pos, 0, AVE_INTER_POS_RESOL * sizeof(float));

  //Initialize random number generator
  inits_GPU<<<N_BLOCKS, N_THREADS>>>(d_state, d_seed);
  check_cuda_errors(__FILE__, __LINE__);

  //Initialize the desired beam pattern, angle sweeps, iteration number
  firefly_init_GPU<<<N_BLOCKS, N_THREADS>>>(d_FFdes, d_cos_phi_sweep, d_sin_phi_sweep, d_sin_theta_sweep, d_iteration_number);
  check_cuda_errors(__FILE__, __LINE__);

  ///////////////////////////////
  //END GPU INITIALIZATION


  //OPEN FILES
  openfiles(&filed, &filepos, &fileavepos, &fileinterpos, &fileaveinterpos, &filedes, &filemean, &filevar);

  filex = fopen("../data/x.txt", "w");
  check_file(&filex);

 filesolncount = fopen("../data/solncount.txt", "w");
 check_file(&filesolncount);

 fileconvergence = fopen("../data/convergence.txt", "w");
 check_file(&fileconvergence);

  //BEGIN
  for(ensemble_number = 0; ensemble_number < ensembles; ensemble_number++){
    if((ensemble_number % 10 == 0) && (ensemble_number != 0)){
      printf("ensemble: %d\n", ensemble_number);
    }

    //Initialize the element positions on the GPU
    positions_init_GPU<<<N_BLOCKS, N_THREADS>>>(d_x, d_state, d_iteration_number);
    check_cuda_errors(__FILE__, __LINE__);

    //Firefly Algorithm
    for(i = 0; i < iterations; i++){

      //Compute the rad pattern
      firefly_radpat2D_GPU<<<N_BLOCKS, N_THREADS>>>(d_x, d_FF, d_cos_phi_sweep, d_sin_phi_sweep, d_sin_theta_sweep, d_khat);
      check_cuda_errors(__FILE__, __LINE__);

      //Compute the fitness
      cudaMemset(d_fitness, 0, FIREFLYNUM * sizeof(float));
      firefly_fitness_GPU<<<N_BLOCKS, N_THREADS>>>(d_FF, d_FFdes, d_fitness);
      check_cuda_errors(__FILE__, __LINE__);

      //Normalize the fitness
      firefly_fitness_norm_GPU<<<1, FIREFLYNUM>>>(d_fitness);
      check_cuda_errors(__FILE__, __LINE__);

      //Copy fitness back so I can do convergence rate analysis
    cudaMemcpy(fitness, d_fitness, FIREFLYNUM * sizeof(float), cudaMemcpyDeviceToHost);
      for(j = 0; j < FIREFLYNUM; j++){
	ave_fitness[i] += fitness[j];
      }

      //Sort the elements
      //First compute distance to origin, r_o
      firefly_r_o_GPU<<<N_BLOCKS, N_THREADS>>>(d_x, d_r_o);
      check_cuda_errors(__FILE__, __LINE__);

      //Now sort via radix block sorting
      firefly_sort_x_GPU<<<N_BLOCKS, NUM_THREADS>>>(d_x, d_r_o);
      check_cuda_errors(__FILE__, __LINE__);

      //Store previous element positions
      //I can probably do this by just copying x array into the x_prev array
      //cudaMemcpy(d_x_prev, d_x, 2 * MY * MX * FIREFLYNUM, cudaMemcpyDevicetoDevice);
      firefly_x_prev_GPU<<<N_BLOCKS, N_THREADS>>>(d_x, d_x_prev);
      check_cuda_errors(__FILE__, __LINE__);

      //First compute beta values
      cudaMemset(d_r, 0, FIREFLYNUM * FIREFLYNUM * sizeof(float));
      firefly_beta_GPU<<<N_BLOCKS, N_THREADS>>>(d_x, d_beta, d_r);
      check_cuda_errors(__FILE__, __LINE__);

      //Normalize beta
      firefly_beta_norm_GPU<<<N_BLOCKS, N_THREADS>>>(d_beta, d_r, d_gamma);
      check_cuda_errors(__FILE__, __LINE__);

      //Then do convergence
      firefly_converge_GPU<<<N_BLOCKS, N_THREADS>>>(d_x, d_x_prev, d_beta, d_fitness);
      check_cuda_errors(__FILE__, __LINE__);

      //Now do random movement
      firefly_random_GPU<<<N_BLOCKS, N_THREADS>>>(d_x, d_alpha0, d_alpha, d_state);
      check_cuda_errors(__FILE__, __LINE__);

      //Update iteration/alpha
      firefly_random_norm_GPU<<<1, 1>>>(d_alpha0, d_alpha, d_iteration_number, iterations);
      check_cuda_errors(__FILE__, __LINE__);
    } //ITERATIONS

    //Find FF, fitness after the last iteration
    //This way we can do the statistics over ensembles
    firefly_radpat2D_GPU<<<N_BLOCKS, N_THREADS>>>(d_x, d_FF, d_cos_phi_sweep, d_sin_phi_sweep, d_sin_theta_sweep, d_khat);
    check_cuda_errors(__FILE__, __LINE__);
    cudaMemset(d_fitness, 0, FIREFLYNUM * sizeof(float));
    firefly_fitness_GPU<<<N_BLOCKS, N_THREADS>>>(d_FF, d_FFdes, d_fitness);
    check_cuda_errors(__FILE__, __LINE__);
    firefly_fitness_norm_GPU<<<1, FIREFLYNUM>>>(d_fitness);
    check_cuda_errors(__FILE__, __LINE__);

    //Statistics
    //See function descriptions for details on how this is done
    moments_counter_GPU<<<1, FIREFLYNUM>>>(d_fitness, d_mean_counter);
    check_cuda_errors(__FILE__, __LINE__);
    moments_GPU<<<N_BLOCKS, N_THREADS>>>(d_FF, d_fitness, d_delta, d_mean, d_M2, d_mean_counter);
    check_cuda_errors(__FILE__, __LINE__);

    //Histograms
    //Sorted first
    firefly_sort_x_GPU<<<N_BLOCKS, NUM_THREADS>>>(d_x, d_r_o);
    check_cuda_errors(__FILE__, __LINE__);

    //Element positions
    pos_hist_GPU<<<N_BLOCKS, N_THREADS>>>(d_x, d_pos, d_fitness);
    check_cuda_errors(__FILE__, __LINE__);
    //Average element position
    ave_pos_hist_GPU<<<N_BLOCKS, N_THREADS>>>(d_x, d_ave_pos, d_fitness);
    check_cuda_errors(__FILE__, __LINE__);
    //Element separation
    inter_pos_hist_GPU<<<N_BLOCKS, N_THREADS>>>(d_x, d_inter_pos, d_fitness);
    check_cuda_errors(__FILE__, __LINE__);
    //Average element separation
    ave_inter_pos_hist_GPU<<<N_BLOCKS, N_THREADS>>>(d_x, d_ave_inter_pos, d_fitness);
    check_cuda_errors(__FILE__, __LINE__);

    //Want to print x array to file
    //Can do correlation and other things with it later
    cudaMemcpy(x, d_x, FIREFLYNUM * ELEMENTNUM * 2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(fitness, d_fitness, FIREFLYNUM * sizeof(float), cudaMemcpyDeviceToHost);
    junk = 0;
    for(i = 0; i < FIREFLYNUM; i++){
      if(fitness[i] >= (0 - epsilon)){  
	junk++;  //if you want soln count
	for(j = 0; j < ELEMENTNUM * 2; j++){
	  fprintf(filex, "%f ", x[i * ELEMENTNUM * 2 + j]);
	}
	fprintf(filex, "\n");
	//break;  //IMPORTANT: ONLY PRINT ONE SOLUTION PER ENSEMBLE
      }
    }
    fprintf(filesolncount, "%f\n", junk);  //If you want soln count
    
  }  //END ENSEMBLES

  //Compile the individual mean/var into a total mean/var
  moments_total_GPU<<<N_BLOCKS, N_THREADS>>>(d_mean, d_mean_total, d_M2, d_M2_total, d_mean_counter, d_mean_counter_total);
  check_cuda_errors(__FILE__, __LINE__);

  //Normalize the mean/var
  moments_total_norm_GPU<<<N_BLOCKS, N_THREADS>>>(d_mean_total, d_M2_total, d_mean_counter_total);
  check_cuda_errors(__FILE__, __LINE__);

  //Move statistics back over
  cudaMemcpy(&mean_counter, d_mean_counter_total, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(mean, d_mean_total, PHI_RESOL * THETA_RESOL * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(M2, d_M2_total, PHI_RESOL * THETA_RESOL * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(pos, d_pos, POS_RESOL * POS_RESOL * ELEMENTNUM * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(ave_pos, d_ave_pos, AVE_POS_RESOL * AVE_POS_RESOL * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(inter_pos, d_inter_pos, INTER_POS_RESOL * ELEMENTNUM * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(ave_inter_pos, d_ave_inter_pos, AVE_INTER_POS_RESOL * sizeof(float), cudaMemcpyDeviceToHost);
  //Faster to copy back or just generate here?  Doesn't even take a cos
  cudaMemcpy(FFdes, d_FFdes, PHI_RESOL * THETA_RESOL * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(fitness, d_fitness, FIREFLYNUM * sizeof(float), cudaMemcpyDeviceToHost);  

  //PRINTING

  //Print ave fitness across ensembles as a function of iterations
  for(i = 0; i < iterations; i++){
    fprintf(fileconvergence, "%d %f\n", i, ave_fitness[i] / ensembles / FIREFLYNUM);
  }

  //First convert all hists to pdfs using trapezoidal/double trapezoidal

  for(i = 0; i < ELEMENTNUM; i++){
    //double_trapezoidal expects RESOL * RESOL amounts of data
    //Pass the address starting each element's subset of data
    temp = double_trapezoidal(POS_RESOL, 0, Lx, &pos[i * POS_RESOL * POS_RESOL]);
    for(j = 0; j < POS_RESOL; j++){
      for(k = 0; k < POS_RESOL; k++){
	element = offset3D(k, j, i, POS_RESOL, POS_RESOL);
	pos[element] = pos[element] / temp;
      }
    }
  }

  temp = double_trapezoidal(AVE_POS_RESOL, 0, Lx, ave_pos);
  for(i = 0; i < AVE_POS_RESOL; i++){
    for(j = 0; j < AVE_POS_RESOL; j++){
      ave_pos[i * AVE_POS_RESOL + j] = ave_pos[i * AVE_POS_RESOL + j] / temp;
    }
  }

  for(i = 0; i < ELEMENTNUM; i++){
    temp = trapezoidal(INTER_POS_RESOL, 0, Lx, &inter_pos[i * INTER_POS_RESOL]);
    for(j = 0; j < INTER_POS_RESOL; j++){
      element = offset2D(j, i, INTER_POS_RESOL);
      inter_pos[element] = inter_pos[element] / temp;
    }
  }

  temp = trapezoidal(AVE_INTER_POS_RESOL, 0, Lx, ave_inter_pos);
  for(i = 0; i < AVE_INTER_POS_RESOL; i++){
    ave_inter_pos[i] = ave_inter_pos[i] / temp;
  }
  
  //Print all the parameters first
  printparams(&filed, &filepos, &fileavepos, &fileinterpos, &fileaveinterpos, &filedes, &filemean, &filevar, alpha0, iterations);
  //Then print data
  printdata(&filed, &filepos, &fileavepos, &fileinterpos, &fileaveinterpos, &filedes, &filemean, &filevar, mean, FFdes, M2, mean_counter, pos, ave_pos, inter_pos, ave_inter_pos);
  
  //CLOSE FILES
  fclose(filepos);
  fclose(fileavepos);
  fclose(fileinterpos);
  fclose(fileaveinterpos);
  fclose(filed);
  fclose(filedes);
  fclose(filemean);
  fclose(filevar);

  //FREE GPU
  cudaFree(d_x);
  cudaFree(d_x_prev);
  cudaFree(d_FF);
  cudaFree(d_FFdes);
  cudaFree(d_cos_phi_sweep);
  cudaFree(d_sin_phi_sweep);
  cudaFree(d_sin_theta_sweep);
  cudaFree(d_beta);
  cudaFree(d_fitness);
  cudaFree(d_r);
  cudaFree(d_gamma);
  cudaFree(d_alpha0);
  cudaFree(d_alpha);
  cudaFree(d_iteration_number);
  cudaFree(d_seed);

  //End timing
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&t, start, stop);
  printf("Time excluding initialization: %3.3f ms \n", t);

  //Return fitness of the first array
  //Not needed or used currently, anyways
  //In the past I returned the fitness of the mean FF
  return(fitness[0]);

}

//This function initializes the seeds for threads
__global__ void inits_GPU(curandState *state, unsigned long *seed){

  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  //Only init for those threads that will need random numbers
  //This is extremely important as not following this can have a huge slowdown
  if(idx < FIREFLYNUM * ELEMENTNUM * 2){
    #ifdef DEBUG
    curand_init(1337, idx, 0, &state[idx]);
    #else
    curand_init(*seed, idx, 0, &state[idx]);
    #endif
  }

  #ifdef DEBUG
  if(idx == 0){
    printf("inits_GPU\n");
  }
  #endif

}

//Initialize the needed arrays on the GPU for the ensembles
//Create FFdes and the phi/theta sweeps
//Initialize the iteration un
__global__ void firefly_init_GPU(float *FFdes, float *cos_phi_sweep, float *sin_phi_sweep, float *sin_theta_sweep, int *iteration_number){

  int i;

  int idx;
  idx = threadIdx.x + blockIdx.x * blockDim.x;

  int dim_elements[2];
  //dim_elements[0] = phi
  //dim_elements[1] = theta
  int dimensions[2];

  dimensions[0] = PHI_RESOL;
  dimensions[1] = THETA_RESOL;

  //Ensure we cover every element (and don't go over)
  //idx operating over range of FFdes
  for(i = idx; i < PHI_RESOL * THETA_RESOL; i = i + N_BLOCKS * N_THREADS){

    //Retrieve dimensions
    dim_elements_GPU(2, dimensions, dim_elements, i);

    //Desired radiation pattern
    //Uniform in phi, changes in theta since theta is now angle of interest
    //Flat SLLdes
    if(dim_elements[1] >= floor(((BWdes / 2) * THETA_RESOL / pi))){
      FFdes[i] = SLLdes;
    }

    //Calculate phi/theta sweeps
    //[0:pio2]
    if(i < PHI_RESOL){
      cos_phi_sweep[i] = cos(i * pio2 / PHI_RESOL);
      sin_phi_sweep[i] = sin(i * pio2 / PHI_RESOL);
    }
    if(i < THETA_RESOL){
      sin_theta_sweep[i] = sin(i * pio2 / THETA_RESOL);
    }
  }

  //Set the iteration number to 0
  if(idx == 0){
    *iteration_number = 0;
  }

  #ifdef DEBUG
  if(idx == 0){
    printf("init_GPU\n");
  }
  #endif

}

//Sorts the elements based on distance to the origin, stored in r_o
__global__ void firefly_r_o_GPU(float *x, float *r_o){

  int i;
  int element;

  int idx;
  idx = threadIdx.x + blockIdx.x * blockDim.x;

  //Operating over r_o
  for(i = idx; i < FIREFLYNUM * ELEMENTNUM; i += N_BLOCKS * N_THREADS){

    //I don't need the sqrt since if x < y then x^2 < y^2
    r_o[i] = x[i * 2] * x[i * 2] + x[i * 2 + 1] * x[i * 2 + 1];
  }

  #ifdef DEBUG
  if(idx == 0){
    printf("r_o_GPU\n");
  }
  #endif

}

//Key sort the elements based on distance to origin
//That is, sort r_o and gather x and y based on that sorting
//I perform two keysorts instead of utilizing the gather function
__global__ void firefly_sort_x_GPU(float *x, float *r_o){

  int i,j;
  
  int idx;
  idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Specialize BlockRadixSort for a 1D block of NUM threads owning ITEMS integer keys each
  typedef cub::BlockRadixSort<float, NUM_THREADS, ITEMS_PER_THREAD, float> BlockRadixSort;

  // Allocate shared memory for BlockRadixSort
  __shared__ typename BlockRadixSort::TempStorage temp_storage;

  // Obtain a segment of consecutive items that are blocked across threads
  float thread_keys[ITEMS_PER_THREAD];
  float thread_keys_2[ITEMS_PER_THREAD];
  float thread_values_x[ITEMS_PER_THREAD];
  float thread_values_y[ITEMS_PER_THREAD];

  //Need to loop in case number of blocks < number of arrays to sort
  for(i = blockIdx.x; i < FIREFLYNUM; i += N_BLOCKS){

    for(j = 0; j < ITEMS_PER_THREAD; j++){
      //Put the keys into the thread specific sub-arrays
      thread_keys[j] = r_o[i * ELEMENTNUM + threadIdx.x * ITEMS_PER_THREAD + j];
      thread_keys_2[j] = thread_keys[j];

      //Now put the values into these thread specific sub-arrays
      //Offset by firefly, offset by items and thread, then index by j item
      //Within one firefly a thread will consider items numbers of elements
      //Times two since there are two dimensions (x,y)
      thread_values_x[j] = x[i * ELEMENTNUM * 2 + threadIdx.x * ITEMS_PER_THREAD * 2 + j * 2];
      thread_values_y[j] = x[i * ELEMENTNUM * 2 + threadIdx.x * ITEMS_PER_THREAD * 2 + j * 2 + 1];
    }

    // Collectively sort the keys
    BlockRadixSort(temp_storage).Sort(thread_keys, thread_values_x);
    BlockRadixSort(temp_storage).Sort(thread_keys_2, thread_values_y);

    //Now put them back into global
    for(j = 0; j < ITEMS_PER_THREAD; j++){
      x[i * ELEMENTNUM * 2 + threadIdx.x * ITEMS_PER_THREAD * 2 + j * 2] = thread_values_x[j];
      x[i * ELEMENTNUM * 2 + threadIdx.x * ITEMS_PER_THREAD * 2 + j * 2 + 1] = thread_values_y[j];
    }

    //Need to sync threads within block before looping over arrays to sort
    __syncthreads();

  }

  #ifdef DEBUG
  if(idx == 0){
    printf("sort_x_GPU\n");
  }
  #endif

}

//Stores the previous positions of the elements so that convergence can occur
//without update schedule confliction
//This way each time step is fixed and movement is not a combination of current
//and past time step
//SYNCHRONOUS
__global__ void firefly_x_prev_GPU(float *x, float *x_prev){

  int i;

  int idx;
  idx = threadIdx.x + blockIdx.x * blockDim.x;

  for(i = idx; i < FIREFLYNUM * ELEMENTNUM * 2; i = i + N_BLOCKS * N_THREADS){
    x_prev[i] = x[i];
  }

  #ifdef DEBUG
  if(idx == 0){
    printf("x_prev_GPU\n");
  }
  #endif
}

//This function computes beta for all fireflies
//If this takes too many register have each thread do x and y position
__global__ void firefly_beta_GPU(float *x, float *beta, float *r){

  int i, j;
  int element, element2;

  int idx;
  idx = threadIdx.x + blockIdx.x * blockDim.x;

  int dim_elements[3];
  //dim_elements[0] = ff_num
  //dim_elements[1] = m_num
  //dim_elements[2] = x,y (2)
  //__shared__ int dimensions[3];
  int dimensions[3];

  dimensions[0] = FIREFLYNUM;
  dimensions[1] = ELEMENTNUM;
  dimensions[2] = 2;
  /*  if(threadIdx.x == 0){
    dimensions[0] = FIREFLYNUM;
    dimensions[1] = M;
    dimensions[2] = 2;
  }
  __syncthreads();*/

  //Ensure we cover every element (and don't go over)
  //idx operating over range of x
  for(i = idx; i < FIREFLYNUM * ELEMENTNUM * 2; i = i + N_BLOCKS * N_THREADS){

    //Retrieve dimensions
    dim_elements_GPU(3, dimensions, dim_elements, i);

    //Determine attractiveness to other fireflies
    for(j = 0; j < FIREFLYNUM; j++){
      if(j != dim_elements[0]){
	//Position between elements of fireflies
	//Using squared Euclidean distance
	element = offset3D_GPU(dim_elements[2], dim_elements[1], j, 2, ELEMENTNUM);
	//Have to store every firefly's distance to every other firefly
	//in order to parallelize convergence
	element2 = offset2D_GPU(j, dim_elements[0], FIREFLYNUM);
	atomicAdd(&(r[element2]), (x[i] - x[element]) * (x[i] - x[element]));
      }
    }
  }

  #ifdef DEBUG
  if(idx == 0){
    printf("beta_GPU\n");
  }
  #endif

}

//This function normalizes beta
__global__ void firefly_beta_norm_GPU(float *beta, float *r, float *gamma){

  int i;

  int idx;
  idx = threadIdx.x + blockIdx.x * blockDim.x;

  for(i = idx; i < FIREFLYNUM * FIREFLYNUM; i = i + N_BLOCKS * N_THREADS){
    beta[i] = beta0 * exp((-1 / (*gamma)) * r[i]);
  }

  #ifdef DEBUG
  if(idx == 0){
    printf("norm_GPU\n");
  }
  #endif

}

//This function handles the convergences towards brighter fireflies
//Uses saved previous positions of all elements to avoid time step confliction
__global__ void firefly_converge_GPU(float *x, float *x_prev, float *beta, float *fitness){

  int i, j;
  int element, element2, element3;

  int idx;
  idx = threadIdx.x + blockIdx.x * blockDim.x;

  int dim_elements[3];
  //dim_elements[0] = ff_num
  //dim_elements[1] = m_num
  //dim_elements[2] = x,y (2)
  int dimensions[3];

  dimensions[0] = FIREFLYNUM;
  dimensions[1] = ELEMENTNUM;
  dimensions[2] = 2;

  //Ensure we cover every element (and don't go over)
  //idx operating over range of x
  for(i = idx; i < FIREFLYNUM * ELEMENTNUM * 2; i = i + N_BLOCKS * N_THREADS){
    
    //Retrieve dimensions
    dim_elements_GPU(3, dimensions, dim_elements, i);
 
    //Need to move towards every other firefly
    for(j = 0; j < FIREFLYNUM; j++){
      if(fitness[j] > fitness[dim_elements[0]]){
	//Only move towards the same element (x or y) as the one we're looking at (i)
	element = offset3D_GPU(dim_elements[2], dim_elements[1], j, 2, ELEMENTNUM);
	element2 = offset2D_GPU(j, dim_elements[0], FIREFLYNUM);
	x[i] = x[i] + beta[element2] * (x_prev[element] - x[i]);
      }
    }
  }

  #ifdef DEBUG
  if(idx == 0){
    printf("converge_GPU\n");
  }
  #endif

}

//This function determines the two dimensional radiation pattern
__global__ void firefly_radpat2D_GPU(float *x, float *FF, float *cos_phi_sweep, float *sin_phi_sweep, float *sin_theta_sweep, float *khat){

  int idx;
  idx = threadIdx.x + blockIdx.x * blockDim.x;

  int i, j, k;
  int element;
  float result;

  int dim_elements[3];
  //dim_elements[0] = ff_num
  //dim_elements[1] = phi_num
  //dim_elements[2] = theta_num
  int dimensions[3];

  dimensions[0] = FIREFLYNUM;
  dimensions[1] = PHI_RESOL;
  dimensions[2] = THETA_RESOL;

  //Ensure we cover every element (and don't go over)
  //idx operating over range of F
  for(i = idx; i < FIREFLYNUM * PHI_RESOL * THETA_RESOL; i = i + N_BLOCKS * N_THREADS){

    //Retrieve dimensions
    dim_elements_GPU(3, dimensions, dim_elements, i);

    //Each thread will do one element of FF
    //Each element of FF uses the entirety of x
    result = 0;
    for(j = 0; j < ELEMENTNUM; j++){
      element = offset3D_GPU(0, j, dim_elements[0], 2, ELEMENTNUM);
      result = result + cos(*khat * x[element] * cos_phi_sweep[dim_elements[1]] * sin_theta_sweep[dim_elements[2]]) * cos(*khat * x[element + 1] * sin_phi_sweep[dim_elements[1]] * sin_theta_sweep[dim_elements[2]]);
    }
    //Normalize
    FF[i] = result / ELEMENTNUM;
  }

  #ifdef DEBUG
  if(idx == 0){
    printf("radpat_GPU\n");
  }
  #endif

}

//This function computes the fitness
__global__ void firefly_fitness_GPU(float *FF, float *FFdes, float *fitness){

  int i, j;
  int element;

  int idx;
  idx = threadIdx.x + blockIdx.x * blockDim.x;

  int dim_elements[3];
  //dim_elements[0] = ff_num
  //dim_elements[1] = phi_num
  //dim_elements[2] = theta_num
  int dimensions[3];

  dimensions[0] = FIREFLYNUM;
  dimensions[1] = PHI_RESOL;
  dimensions[2] = THETA_RESOL;

  //Ensure we cover every element (and don't go over)
  //idx operating over range of FF
  for(i = idx; i < FIREFLYNUM * PHI_RESOL * THETA_RESOL; i = i + N_BLOCKS * N_THREADS){

    //Retrieve dimensions
    dim_elements_GPU(3, dimensions, dim_elements, i);

    element = offset2D_GPU(dim_elements[2], dim_elements[1], THETA_RESOL);

    if(20 * log(fabs(FF[i])) > FFdes[element]){
      atomicAdd(&(fitness[dim_elements[0]]), \
		(20 * log(fabs(FF[i])) - FFdes[element]) * (20 * log(fabs(FF[i])) - FFdes[element]));
    }
  }

  #ifdef DEBUG
  if(idx == 0){
    printf("fitness_GPU\n");
  }
  #endif
  
}

//This function normalizes fitness
__global__ void firefly_fitness_norm_GPU(float *fitness){

  int i;

  int idx;
  idx = threadIdx.x + blockIdx.x * blockDim.x;

  //Ensure no accidental OOB of array
  if(idx < FIREFLYNUM){
    //Normalize with frequency resolution so higher PHI_RESOL != higher fcost
    //Also make negative since my version of "fitness" is actually cost
    fitness[idx] = -fitness[idx] / PHI_RESOL / THETA_RESOL;
  }

  #ifdef DEBUG
  if(idx == 0){
    printf("fitnessnorm_GPU\n");
    for(i = 0; i < FIREFLYNUM; i++){
      printf("fitness[%d]: %f\n", i, fitness[i]);
    }
  }
  #endif
  
}
//First compute the mean_counters
__global__ void moments_counter_GPU(float *fitness, int *mean_counter){

  int idx;
  idx = threadIdx.x + blockIdx.x * blockDim.x;

  int i;

  //Ensure we cover every element (and don't go over)
  //idx operating over range of FF/delta
  for(i = idx; i < FIREFLYNUM; i = i + N_BLOCKS * N_THREADS){
    
    //Only compute for those that possess desired fitness
    if(fitness[i] >= (0 - epsilon)){
      atomicAdd(&(mean_counter[i]), 1);
    }
  }

  #ifdef DEBUG
  if(idx == 0){
    printf("moments_GPU\n");
  }
  #endif

}
//Calculate the mean/variance
//Online statistic
//Done for each firefly and only combined after ensembles (horiz, then vert)
__global__ void moments_GPU(float *FF, float *fitness, float *delta, float *mean, float *M2, int *mean_counter){

  int idx;
  idx = threadIdx.x + blockIdx.x * blockDim.x;

  int i;
  int element;

  int dim_elements[3];
  //dim_elements[0] = ff_num
  //dim_elements[1] = phi_num
  //dim_elements[2] = theta_num
  int dimensions[3];
  
  dimensions[0] = FIREFLYNUM;
  dimensions[1] = PHI_RESOL;
  dimensions[2] = THETA_RESOL;

  //Ensure we cover every element (and don't go over)
  //idx operating over range of FF/delta
  for(i = idx; i < FIREFLYNUM * PHI_RESOL * THETA_RESOL; i = i + N_BLOCKS * N_THREADS){

    //Retrieve dimensions
    dim_elements_GPU(3, dimensions, dim_elements, i);

    //Only compute for those that possess desired fitness
    if(fitness[dim_elements[0]] >= (0 - epsilon)){

      //Now do online algorithm
      //See Knuth
      delta[i] = FF[i] - mean[i];
      mean[i] = mean[i] + delta[i] / mean_counter[dim_elements[0]];  //Mean
      M2[i] = M2[i] + delta[i] * (FF[i] - mean[i]);  //Sum of squares
    }
  }

  #ifdef DEBUG
  if(idx == 0){
    printf("moments_GPU\n");
  }
  #endif

}


//Combine the results of mean/var now that ensembles are done
__global__ void moments_total_GPU(float *mean, float *mean_total, float *M2, float *M2_total, int *mean_counter, int *mean_counter_total){

  int idx;
  idx = threadIdx.x + blockIdx.x * blockDim.x;

  int i;
  int element;

  int dim_elements[3];
  //dim_elements[0] = ff_num
  //dim_elements[1] = phi_num
  //dim_elements[2] = theta_num
  int dimensions[3];
  
  dimensions[0] = FIREFLYNUM;
  dimensions[1] = PHI_RESOL;
  dimensions[2] = THETA_RESOL;

  //Ensure we cover every element (and don't go over)
  //idx operating over range of mean
  for(i = idx; i < FIREFLYNUM * PHI_RESOL * THETA_RESOL; i = i + N_BLOCKS * N_THREADS){

    //Retrieve dimensions
    dim_elements_GPU(3, dimensions, dim_elements, i);
    element = offset2D_GPU(dim_elements[2], dim_elements[1], THETA_RESOL);

    //Have to make sure there's something to add
    //If mean_counter is 0 it craps out and M2_total goes to nan
    //No idea why.  Maybe atomicAdd( ,0) produces nan because 0 is not int/float?
    if(mean_counter[dim_elements[0]] > 0){
      atomicAdd(&(mean_total[element]), mean_counter[dim_elements[0]] * mean[i]);
      atomicAdd(&(M2_total[element]), mean_counter[dim_elements[0]] * (M2[i] / mean_counter[dim_elements[0]] + mean[i] * mean[i]));

      //Make sure we only update mean_counter_total once per firefly
      if(dim_elements[1] == 0 && dim_elements[2] == 0){
	atomicAdd(mean_counter_total, mean_counter[dim_elements[0]]);
      }
    }
  }

  #ifdef DEBUG
  if(idx == 0){
    printf("moments_total_GPU\n");
  }
  #endif

}


//Normalize by total number of samples
__global__ void moments_total_norm_GPU(float *mean_total, float *M2_total, int *mean_counter_total){

  int idx;
  idx = threadIdx.x + blockIdx.x * blockDim.x;

  int i;
  int element;

  //Ensure we cover every element (and don't go over)
  //idx operating over range of mean_total
  for(i = idx; i < PHI_RESOL * THETA_RESOL; i = i + N_BLOCKS * N_THREADS){

    mean_total[i] = mean_total[i] / *mean_counter_total;
    //M2 is now variance instead of sd
    M2_total[i] = M2_total[i] / *mean_counter_total - mean_total[i] * mean_total[i];
  }

  #ifdef DEBUG
  if(idx == 0){
    printf("moments_total_norm_GPU\n");
  }
  #endif

}


//CPU FUNCTIONS
///////////////////////////////////
/*
//radpat2D computes the radiation pattern based on the equation of symmetrical
//elements about the x,y-axis.
void radpat2D(float FF[],float d[], float khat){

  int i, j, k, l, m;
  int element, element2;
  float d_xy[2];
  
  for(i = 0; i < PHI_RESOL; i++){
    for(j = 0; j < THETA_RESOL; j++){
      element = offset2D(j, i, THETA_RESOL);
      for(k = 0; k < M; k++){
	for(l = 0; l < MY; l++){
	  //Have to get the elements of the x,y positions	
	  for(m = 0; m < 2; m++){
	    element2 = offset3D(m, l, k, 2, MY);
	    d_xy[m] = d[element2];
	  }
	  //Now the element of FF
	  FF[element] = FF[element] + cos(khat * d_xy[0] * cos_phi_sweep[i] * sin_theta_sweep[j]) * cos(khat * d_xy[1] * sin_phi_sweep[i] * sin_theta_sweep[j]);
	}
      }
      //Normalization
      FF[element] = FF[element] / (MX * MY);
    }
  }
}


//fcost2D determines the fitness of the firefly based on its beamform and the
//desired beamform using a simple (x-hat(x))^2.
//This is in dB
float fcost2D(float *FF,float FFdes[]){

  int i, j;
  int element;
  float f = 0;

  //For some reason I handle the first element separately
  //I'm not sure why but don't feel like investigating atm
  //Main difference seems to be no comparison in dB
  //Perhaps FF[0] was usually zero
  //if(FF[0] > FFdes[0]){
  //  f = f + (20 * log(fabs(FF[0])) - FFdes[0]) * (20 * log(fabs(FF[0])) - FFdes[0]);
  //}
  for(i = 0; i < PHI_RESOL; i++){
    //    for(j = 0; j < THETA_RESOL; j++){
    for(j = 0; j < THETA_RESOL; j++){
      //      if((i != 0) && (j != 0)){
	element = offset2D(j, i, THETA_RESOL);
	if(20 * log(fabs(FF[element])) > FFdes[element]){
	  f = f + (20 * log(fabs(FF[element])) - FFdes[element]) * (20 * log(fabs(FF[element])) - FFdes[element]);
	}
	//      }
    }
  }
  //Normalize with frequency resolution so higher PHI_RESOL != higher fcost
  f = f / PHI_RESOL / THETA_RESOL;
  return(f);
}
*/
