#include "nicklib_GPU.h"

//These functions determine element offset for unrolled 2D/3D matrices
__device__ int offset2D_GPU(int dim1, int dim2, int dim1Size){
  return((dim2 * dim1Size) + dim1);
}
__device__ int offset3D_GPU(int dim1, int dim2, int dim3, int dim1Size, int dim2Size){
  return((dim3 * dim1Size * dim2Size) + (dim2 * dim1Size) + dim1);
}
__device__ int offset4D_GPU(int dim1, int dim2, int dim3, int dim4, int dim1Size, int dim2Size, int dim3Size){
  return((dim4 * dim1Size * dim2Size * dim3Size) + (dim3 * dim1Size * dim2Size) + (dim2 * dim1Size) + dim1);
}

//Calculates the element of each rolled up dimension given the element of
//the unrolled array.
//floor((idx % inner_product from i > N - 1) / outer_product from i + 1 to N - 1)
//Last dimension has no division
//Returns values in dim_elements
__device__ void dim_elements_GPU(int total_dims, int dimensions[], int dim_elements[], int element){

  int i, j;
  int inner, outer;

  for(i = 0; i < total_dims; i++){  //Do every dimension
    outer = 1;
    for(j = i; j < total_dims; j++){  //Calculate the outer product
      if(j != i && i < total_dims - 1){  //Could eliminate by changing j loop
	outer = outer * dimensions[j];
      }
    }
    inner = outer * dimensions[i];  //Saves some operations
    if(i < total_dims - 1){  //Do I really need this?  Should  already be handled by j as when i = total_dims - 1, j loop won't go so outer = 1.
      dim_elements[i] = (int)((float)(element % inner) / outer);
    }
    else{
      dim_elements[i] = element % inner;
    }
  }
}
