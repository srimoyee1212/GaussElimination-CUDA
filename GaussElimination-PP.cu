#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "cuda.h"
#include <string.h>

#define MAXBLOCKSIZE 512

int Size;
float *a, *b, *finalVec;
float *m;

FILE *fp;
void ForwardSub();
void BackSub();
void checkCUDAError(const char *msg);

void InitPerRun() 
{
	int i;
	for (i=0; i<Size*Size; i++)
			*(m+i) = 0.0;
}


void InitMat(float *ary, int nrow, int ncol)
{
	int i, j;
	
	for (i=0; i<nrow; i++) {
		for (j=0; j<ncol; j++) {
			fscanf(fp, "%f",  ary+Size*i+j);
		}
	}  
}

void InitAry(float *ary, int ary_size)
{
	int i;
	
	for (i=0; i<ary_size; i++) {
		fscanf(fp, "%f",  &ary[i]);
	}
}

void PrintMat(float *ary, int nrow, int ncol)
{
	int i, j;
	
	for (i=0; i<nrow; i++) {
		for (j=0; j<ncol; j++) {
			printf("%8.2f ", *(ary+Size*i+j));
		}
		printf("\n");
	}
	printf("\n");
}

__global__ void Fan1(float *m_cuda, float *a_cuda, int Size, int t)
{   
	

	if(threadIdx.x + blockIdx.x * blockDim.x >= Size-1-t) return;
	*(m_cuda+Size*(blockDim.x*blockIdx.x+threadIdx.x+t+1)+t) = *(a_cuda+Size*(blockDim.x*blockIdx.x+threadIdx.x+t+1)+t) / *(a_cuda+Size*t+t);
}

__global__ void Fan2(float *m_cuda, float *a_cuda, float *b_cuda,int Size, int j1, int t)
{
	if(threadIdx.x + blockIdx.x * blockDim.x >= Size-1-t) return;
	if(threadIdx.y + blockIdx.y * blockDim.y >= Size-t) return;
	
	int xidx = blockIdx.x * blockDim.x + threadIdx.x;
	int yidx = blockIdx.y * blockDim.y + threadIdx.y;
	
	
	a_cuda[Size*(xidx+1+t)+(yidx+t)] -= m_cuda[Size*(xidx+1+t)+t] * a_cuda[Size*t+(yidx+t)];
	
	if(yidx == 0){
		
		b_cuda[xidx+1+t] -= m_cuda[Size*(xidx+1+t)+(yidx+t)] * b_cuda[t];
	}
}




void ForwardSub()
{
	int t;
    float *m_cuda,*a_cuda,*b_cuda;
	
	
	cudaMalloc((void **) &m_cuda, Size * Size * sizeof(float));
	 
	cudaMalloc((void **) &a_cuda, Size * Size * sizeof(float));
	
	cudaMalloc((void **) &b_cuda, Size * sizeof(float));	

	
	cudaMemcpy(m_cuda, m, Size * Size * sizeof(float),cudaMemcpyHostToDevice );
	cudaMemcpy(a_cuda, a, Size * Size * sizeof(float),cudaMemcpyHostToDevice );
	cudaMemcpy(b_cuda, b, Size * sizeof(float),cudaMemcpyHostToDevice );
	
	int block_size,grid_size;
	
	block_size = MAXBLOCKSIZE;
	grid_size = (Size/block_size) + (!(Size%block_size)? 0:1);
	


	dim3 dimBlock(block_size);
	dim3 dimGrid(grid_size);
	
	
	int blockSize2d, gridSize2d;
	blockSize2d = 4;
	gridSize2d = (Size/blockSize2d) + (!(Size%blockSize2d?0:1)); 
	
	dim3 dimBlockXY(blockSize2d,blockSize2d);
	dim3 dimGridXY(gridSize2d,gridSize2d);

   
    struct timeval time_start;
    gettimeofday(&time_start, NULL);
	for (t=0; t<(Size-1); t++) {
		Fan1<<<dimGrid,dimBlock>>>(m_cuda,a_cuda,Size,t);
		cudaThreadSynchronize();
		Fan2<<<dimGridXY,dimBlockXY>>>(m_cuda,a_cuda,b_cuda,Size,Size-t,t);
		cudaThreadSynchronize();
		checkCUDAError("Fan2");
	}
	
	struct timeval time_end;
    gettimeofday(&time_end, NULL);
   // totalKernelTime = (time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec);
	
	
	cudaMemcpy(m, m_cuda, Size * Size * sizeof(float),cudaMemcpyDeviceToHost );
	cudaMemcpy(a, a_cuda, Size * Size * sizeof(float),cudaMemcpyDeviceToHost );
	cudaMemcpy(b, b_cuda, Size * sizeof(float),cudaMemcpyDeviceToHost );
	cudaFree(m_cuda);
	cudaFree(a_cuda);
	cudaFree(b_cuda);
}

void BackSub()
{
	
	finalVec = (float *) malloc(Size * sizeof(float));
	
	int i,j;
	for(i=0;i<Size;i++){
		finalVec[Size-i-1]=b[Size-i-1];
		for(j=0;j<i;j++)
		{
			finalVec[Size-i-1]-=*(a+Size*(Size-i-1)+(Size-j-1)) * finalVec[Size-j-1];
		}
		finalVec[Size-i-1]=finalVec[Size-i-1]/ *(a+Size*(Size-i-1)+(Size-i-1));
	}
}



void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, 
                                  cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         
}


void PrintAry(float *ary, int ary_size)
{
	int i;
	for (i=0; i<ary_size; i++) {
		printf("%8.2f ", ary[i]);
	}
	printf("\n\n");
}
int main()
{
	fp=fopen("samplege.txt","r");
        fscanf(fp, "%d", &Size);	
	 
	a = (float *) malloc(Size * Size * sizeof(float));
	 
	InitMat(a, Size, Size);
	
	b = (float *) malloc(Size * sizeof(float));
	
	InitAry(b, Size);
	
		
	 m = (float *) malloc(Size * Size * sizeof(float));

	InitPerRun();

        printf("Matrix a is: \n");
        PrintMat(a, Size, Size);
	
	printf("Array b is: \n");
        PrintAry(b, Size);

	
	ForwardSub();
	BackSub();
	printf("The final solution is: \n");
        PrintAry(finalVec,Size);

	
        free(m);
        free(a);
        free(b);
          
}
