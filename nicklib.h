#ifndef NICKLIB_H_
#define NICKLIB_H_

double rand_num(double min, double max);
void gaussian(float gaussian[], float mean, float sd);
void insertion(float A[], int end);
void insertion_multi(int N, int M, float **A);
int offset2D(int dim1, int dim2, int dim1Size);
int offset3D(int dim1, int dim2, int dim3, int dim1Size, int dim2Size);
int offset4D(int dim1, int dim2, int dim3, int dim4, int dim1Size, int dim2Size, int dim3Size);

#endif
