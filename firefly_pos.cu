#include "firefly_pos.h"
#include "firefly.h"
#include "../../../nicklib/nicklib_GPU.h"
extern "C"{
#include "../../../nicklib/nicklib.h"
#include "../run2DCUDA/defines.h"
}
#include <stdio.h>

//This function initializes the element positions for the ensembles
//Resets the iteration number
__global__ void positions_init_GPU(float *x, curandState *state, int *iteration_number){

  int i;
  float value;

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

    //Determine dimensions
    dim_elements_GPU(3, dimensions, dim_elements, i);

    //See notes for algorithm details
    //Combined x,y calculations into a single line that utilizes dim_elements[3]
    //Elements are placed at the center of their boundaries
    x[i] = curand_uniform(&state[idx]) * ((i % 2 == 0) ? Lx : Ly);
  }

  if(idx == 0){
    *iteration_number = 0;
  }

  #ifdef DEBUG
  if(idx == 0){
    printf("pos_GPU\n");
  }
  #endif

}

//This function handles the random movement part
//It is bounded so as not to go OOB
//Upper/lower boundary checks for both x and y
//Linearly decrease alpha
__global__ void firefly_random_GPU(float *x, float *alpha0, float *alpha, curandState *state){

  int i, j;
  float value;

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

    //Determine dimensions
    dim_elements_GPU(3, dimensions, dim_elements, i);

    //Segmented
    //Shape random movement so that there's no OOB
    //Use dim_elements[2] to combine x,y calculations
    
    //Lower boundary
    if(x[i] < *alpha / 2){

      //Uniform (-1, 1)
      value = curand_uniform(&state[idx]) * 2 - 1;
      //Ensure it's zero mean
      //If < 0 scale to boundary
      //If > 0 add directly
      if(value < 0){
	x[i] += value * x[i];
      }
      else{
	x[i] += value * (*alpha / 2);
      }
    }
      //Upper boundary
    else if(x[i] > ((i % 2 == 0 ? Lx : Ly) - *alpha / 2)){
      //In case above ? operator doesn't work, assume Lx = Ly for now
      //else if(x[i] > (Lx - *alpha / 2)){
      value = curand_uniform(&state[idx]) * 2 - 1;
      if(value > 0){
	x[i] += value * ((i % 2 == 0 ? Lx : Ly) - x[i]);
	//x[i] += value * (Lx - x[i]);
      }
      else{
	x[i] += value * (*alpha / 2);
      }
    }
    //Otherwise good to go
    else{
      x[i] += curand_uniform(&state[idx]) * (*alpha) - (*alpha / 2);
    }
  }

  
  #ifdef DEBUG
  if(idx == 0){
    printf("rand_GPU\n");
  }
  #endif
  
}


//This function updates the iteration number/alpha
//Only uses one thread
__global__ void firefly_random_norm_GPU(float *alpha0, float *alpha, int *iteration_number, int iterations){

  int idx;
  idx = threadIdx.x + blockIdx.x * blockDim.x;

  //Do a check to make sure, anyways
  if(idx == 0){
    //Update the iteration number
    *iteration_number = *iteration_number + 1;
    //Decrease alpha linearly to zero over iterations
    *alpha = *alpha0 - *alpha0 * (*iteration_number + 1) / iterations;
  }
    
  #ifdef DEBUG
  printf("rand_norm_GPU\n");
  #endif

}


//Absolute position histogram
//Forms an independent histogram for each element
__global__ void pos_hist_GPU(float *x, float *pos, float *fitness){

  int i;
  int element;

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

  //idx operating over range of x
  //Use even index since we handle x and y simultaneously
  for(i = idx * 2; i < FIREFLYNUM * ELEMENTNUM * 2; i = i + N_BLOCKS * N_THREADS){

      //Determine dimensions
      dim_elements_GPU(3, dimensions, dim_elements, i);
      
      if(fitness[dim_elements[0]] >= (0 - epsilon)){
	
	//First get the x,y positions
	//element = offset3D_GPU(0, dim_elements[1], dim_elements[0], 2, ELEMENTNUM);
	//Now determine which value of hist to update
	//The x position is from temp and the y position is from junk
	//Replace 'i' with 'element' if this doesn't work
       	element = offset3D_GPU(					\
			       (int)(x[i + 1] * ((float)POS_RESOL / Ly)), \
			       (int)(x[i] * ((float)POS_RESOL / Lx)), \
			       dim_elements[1], \
			       POS_RESOL, POS_RESOL);

	//ERROR CHECKING
	#ifdef DEBUG
	if(element >= POS_RESOL * POS_RESOL * ELEMENTNUM){
	  printf("OUT OF BOUNDS OF POS ARRAY!\n");
	  printf("element: %d\n", element);
	}
	#endif

	atomicAdd(&(pos[element]), 1);
      }
      //}
  }

  #ifdef DEBUG
  if(idx == 0){
    printf("pos_hist_GPU\n");
  }
  #endif


}


//Averaged position of an element within the array
__global__ void ave_pos_hist_GPU(float *x, float *ave_pos, float *fitness){

  int i;
  int element;
  float temp, junk;

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

  //idx operating over range of x
  //Use even index since we handle x and y simultaneously
  for(i = idx * 2; i < FIREFLYNUM * ELEMENTNUM * 2; i = i + N_BLOCKS * N_THREADS){

    //Only use even threads
    //Tried to eliminate this by modifying i but it got screwed up
    //Not a big deal to do it this way
    //if(idx % 2 == 0){

      //Determine dimensions
      dim_elements_GPU(3, dimensions, dim_elements, i);
      
      if(fitness[dim_elements[0]] >= (0 - epsilon)){
	
	//First get the x,y positions
	//element = offset3D_GPU(0, dim_elements[1], dim_elements[0], 2, ELEMENTNUM);
	//Now determine which value of hist to update
	//The x position is from temp and the y position is from junk
	//Replace 'i' with 'element' if this doesn't work
	element = offset2D_GPU(						\
			       (int)(x[i + 1] * ((float)AVE_POS_RESOL / Ly)), \
			       (int)(x[i] * ((float)AVE_POS_RESOL / Lx)), \
			       AVE_POS_RESOL);

	//ERROR CHECKING
	#ifdef DEBUG
	if(element >= AVE_POS_RESOL * AVE_POS_RESOL){
	  printf("OUT OF BOUNDS OF AVE_POS ARRAY!\n");
	  printf("element: %d\n", element);
	}
	#endif

	atomicAdd(&(ave_pos[element]), 1);
      }
      //}
  }

  #ifdef DEBUG
  if(idx == 0){
    printf("ave_pos_hist_GPU\n");
  }
  #endif

}


//Inter-element distance computed for both x and y directions
__global__ void inter_pos_hist_GPU(float *x, float *inter_pos, float *fitness){

  int i,j;
  int element, element2;

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

  //idx operating over range of x
  //Use even index since we handle x and y simultaneously
  for(i = idx * 2; i < FIREFLYNUM * ELEMENTNUM * 2; i = i + N_BLOCKS * N_THREADS){

    //Determine which firefly/x_num/y_num we're operating as
    dim_elements_GPU(3, dimensions, dim_elements, i);
    
    if(fitness[dim_elements[0]] >= (0 - epsilon)){
      
      //Take radial squared distance to every other element
      for(j = 0; j < ELEMENTNUM; j++){
	if(j != dimensions[1]){
	  
	  //The other index of x array to consider
	  element = offset3D_GPU(0, j, dimensions[0], 2, ELEMENTNUM);
	  
	  atomicAdd(&(inter_pos[dim_elements[1] * INTER_POS_RESOL + (int)(((x[i + 1] - x[element + 1]) * (x[i + 1] - x[element + 1]) + (x[i] - x[element]) * (x[i] - x[element])) * INTER_POS_RESOL / (Lx * Lx + Ly * Ly))]), 1);
	}
      }
    }
  }
  
  #ifdef DEBUG
  if(idx == 0){
    printf("inter_pos_hist_GPU\n");
  }
  #endif

}



//Average inter-element distance computed for both x and y directions
__global__ void ave_inter_pos_hist_GPU(float *x, float *ave_inter_pos, float *fitness){

  int i,j;
  int element, element2;

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

  //idx operating over range of x
  //Use even index since we handle x and y simultaneously
  for(i = idx * 2; i < FIREFLYNUM * ELEMENTNUM * 2; i = i + N_BLOCKS * N_THREADS){

    //Determine which firefly/x_num/y_num we're operating as
    dim_elements_GPU(3, dimensions, dim_elements, i);
    
    if(fitness[dim_elements[0]] >= (0 - epsilon)){
      
      //Take radial squared distance to every other element
      for(j = 0; j < ELEMENTNUM; j++){
	if(j != dimensions[1]){
	  
	  //The other index of x array to consider
	  element = offset3D_GPU(0, j, dimensions[0], 2, ELEMENTNUM);
	  
	  atomicAdd(&(ave_inter_pos[(int)(((x[i + 1] - x[element + 1]) * (x[i + 1] - x[element + 1]) + (x[i] - x[element]) * (x[i] - x[element])) * AVE_INTER_POS_RESOL / (Lx * Lx + Ly * Ly))]), 1);
	}
      }
    }
  }
  
  #ifdef DEBUG
  if(idx == 0){
    printf("ave_inter_pos_hist_GPU\n");
  }
  #endif

}

