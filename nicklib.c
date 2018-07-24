#include <math.h>
#include <stdlib.h>

//rand_num returns a random number, uniformly drawn, in the range of min and max
double rand_num(double min, double max){

  return( ((double)rand() / ((double)RAND_MAX)) * (max - min) + min );
}
//gaussian returns two gaussian numbers in an array
void gaussian(float gauss[], float mean, float sd){
 
  float uni[2];
  float s;

  uni[0] = rand_num(-1, 1);
  uni[1] = rand_num(-1, 1);
  s = uni[0]*uni[0] + uni[1]*uni[1];
  while( (s == 0) || (s >= 1)){
    uni[0] = rand_num(-1, 1);
    uni[1] = rand_num(-1, 1);
    s = uni[0]*uni[0] + uni[1]*uni[1];
  }
  gauss[0] = uni[0] * sqrt((-2 * log(s))/s);
  gauss[1] = uni[1] * sqrt((-2 * log(s))/s);
  gauss[0] = (gauss[0] * sd) + mean;
  gauss[1] = (gauss[1] * sd) + mean;

}
//Insertion sort
void insertion(float A[],int end){

  int i,j;
  float temp;

  for(i = 1; i < end; i++){
    temp = A[i];
    j = i;
    while( (j > 0) && (A[j-1] > temp)){
      A[j] = A[j-1];
      j -= 1;
    }
    A[j] = temp;
  }
}
//Sorts the first row
//Organizes all other COLUMNS by the sorting done on first COLUMN
void insertion_multi(int N, int M, float **A){

  int i, j, k;
  float *temp = NULL;

  temp = malloc(M * sizeof(float));

  for(i = 1; i < N; i++){
    for(k = 0; k < M; k++){
      temp[k] = A[i][k];
    }
    j = i;
    while( (j > 0) && (A[j-1][0] > temp[0])){
      for(k = 0; k < M; k++){
	A[j][k] = A[j-1][k];
      }
      j -= 1;
    }
    for(k = 0; k < M; k++){
      A[j][k] = temp[k];
    }
  }
  free(temp);
}
//These functions determine element offset for unrolled 2D/3D matrices
int offset2D(int dim1, int dim2, int dim1Size){
  return((dim2 * dim1Size) + dim1);
}
int offset3D(int dim1, int dim2, int dim3, int dim1Size, int dim2Size){
  return((dim3 * dim1Size * dim2Size) + (dim2 * dim1Size) + dim1);
}
int offset4D(int dim1, int dim2, int dim3, int dim4, int dim1Size, int dim2Size, int dim3Size){
  return((dim4 * dim1Size * dim2Size * dim3Size) + (dim3 * dim1Size * dim2Size) + (dim2 * dim1Size) + dim1);
}
