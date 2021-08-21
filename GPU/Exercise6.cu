#include "../common/common.h"
#include <stdio.h>

// Dimensione del blocco
#define BDIMX 16
#define BDIMY 16

// macro x conversione indici lineari
#define INDEX(rows, cols, stride) (rows * stride + cols)

// prototipi funzioni
void initialData(float*, const int);
void printData(float*, const int);
void checkResult(float*, float*, int, int);
void transposeHost(float*, float*, const int, const int);
__global__ void copyGmem(float*, float*, const int, const int);
__global__ void naiveGmem(float*, float*, const int, const int);

/*
 * Kernel per il calcolo della matrice trasposta usando la shared memory
 */
__global__ void transposeSmem(float *out, float *in, int nrows, int ncols) {
	
	 // TODO
}

int main(int argc, char **argv) {
	// set up device
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("%s starting transpose at ", argv[0]);
	printf("device %d: %s ", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	bool iprint = 0;

	// set up array size
	int nrows = 1 << 10;
	int ncols = 1 << 10;

	if (argc > 1)
		iprint = atoi(argv[1]);
	if (argc > 2)
		nrows = atoi(argv[2]);
	if (argc > 3)
		ncols = atoi(argv[3]);

	printf("\nMatrice con nrows = %d ncols = %d\n", nrows, ncols);
	size_t ncells = nrows * ncols;
	size_t nBytes = ncells * sizeof(float);

	// execution configuration
	dim3 block(BDIMX, BDIMY);
	dim3 grid((ncols + block.x - 1) / block.x, (nrows + block.y - 1) / block.y);
	dim3 grid2((grid.x + 2 - 1) / 2, grid.y);

	// allocate host memory
	float *h_A = (float *) malloc(nBytes);
	float *hostRef = (float *) malloc(nBytes);
	float *gpuRef = (float *) malloc(nBytes);

	//  initialize host array
	initialData(h_A, nrows * ncols);

	//  transpose at host side
	transposeHost(hostRef, h_A, nrows, ncols);

	// allocate device memory
	float *d_A, *d_C;
	CHECK(cudaMalloc((float** )&d_A, nBytes));
	CHECK(cudaMalloc((float** )&d_C, nBytes));

	// copy data from host to device
	CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

	// tranpose gmem
	CHECK(cudaMemset(d_C, 0, nBytes));
	memset(gpuRef, 0, nBytes);

	double iStart = seconds();
	copyGmem<<<grid, block>>>(d_C, d_A, nrows, ncols);
	CHECK(cudaDeviceSynchronize());
	double iElaps = seconds() - iStart;
	CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
	if (iprint)
		printData(gpuRef, nrows * ncols);
	float ibnd = 2 * ncells * sizeof(float) / (1024.0 * 1024.0 * 1024.0) / iElaps;
	ibnd = 2 * ncells * sizeof(float) / 1e9 / iElaps;
	printf("copyGmem elapsed %f sec\n <<< grid (%d,%d) block (%d,%d)>>> "
			"effective bandwidth %f GB\n", iElaps, grid.x, grid.y, block.x, block.y, ibnd);

	// tranpose gmem
	CHECK(cudaMemset(d_C, 0, nBytes));
	memset(gpuRef, 0, nBytes);

	iStart = seconds();
	naiveGmem<<<grid, block>>>(d_C, d_A, nrows, ncols);
	CHECK(cudaDeviceSynchronize());
	iElaps = seconds() - iStart;

	CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

	if (iprint)
		printData(gpuRef, ncells);

	checkResult(hostRef, gpuRef, ncols, nrows);
	ibnd = 2 * ncells * sizeof(float) / (1024.0 * 1024.0 * 1024.0) / iElaps;
	ibnd = 2 * ncells * sizeof(float) / 1e9 / iElaps;
	printf("naiveGmem elapsed %f sec\n <<< grid (%d,%d) block (%d,%d)>>> "
			"effective bandwidth %f GB\n", iElaps, grid.x, grid.y, block.x,
			block.y, ibnd);

	// tranpose smem
	CHECK(cudaMemset(d_C, 0, nBytes));
	memset(gpuRef, 0, nBytes);

	iStart = seconds();
	transposeSmem<<<grid, block>>>(d_C, d_A, nrows, ncols);
	CHECK(cudaDeviceSynchronize());
	iElaps = seconds() - iStart;

	CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

	if (iprint)
		printData(gpuRef, ncells);

	checkResult(hostRef, gpuRef, ncols, nrows);
	ibnd = 2 * ncells * sizeof(float) / (1024.0 * 1024.0 * 1024.0) / iElaps;
	ibnd = 2 * ncells * sizeof(float) / 1e9 / iElaps;
	printf("transposeSmem elapsed %f sec\n <<< grid (%d,%d) block (%d,%d)>>> "
			"effective bandwidth %f GB\n", iElaps, grid.x, grid.y, block.x,
			block.y, ibnd);

	// free host and device memory
	CHECK(cudaFree(d_A));
	CHECK(cudaFree(d_C));
	free(h_A);
	free(hostRef);
	free(gpuRef);

	// reset device
	CHECK(cudaDeviceReset());
	return EXIT_SUCCESS;
}

void initialData(float *in, const int size) {
	for (int i = 0; i < size; i++)
		in[i] = i; // (float)(rand()/INT_MAX) * 10.0f;
	return;
}

void printData(float *in, const int size) {
	for (int i = 0; i < size; i++)
		printf("%3.0f ", in[i]);
	printf("\n");
	return;
}

void transposeHost(float *out, float *in, const int nrows, const int ncols) {
	for (int iy = 0; iy < nrows; ++iy)
		for (int ix = 0; ix < ncols; ++ix)
			out[INDEX(ix, iy, nrows)] = in[INDEX(iy, ix, ncols)];
}

__global__ void copyGmem(float *out, float *in, const int nrows,
		const int ncols) {
	// matrix coordinate (ix,iy)
	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

	// transpose with boundary test
	if (row < nrows && col < ncols)
		// NOTE this is a transpose, not a copy
		out[INDEX(col, row, nrows)] = in[INDEX(row, col, ncols)];
}

__global__ void naiveGmem(float *out, float *in, const int nrows,
		const int ncols) {
	// matrix coordinate (ix,iy)
	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

	// transpose with boundary test
	if (row < nrows && col < ncols)
		out[INDEX(col, row, nrows)] = in[INDEX(row, col, ncols)];
}

void checkResult(float *hostRef, float *gpuRef, int rows, int cols) {
	double epsilon = 1.0E-8;
	bool match = 1;

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			int index = INDEX(i, j, cols);
			if (abs(hostRef[index] - gpuRef[index]) > epsilon) {
				match = 0;
				printf("different on (%d, %d) (offset=%d) element in "
						"transposed matrix: host %f gpu %f\n", i, j, index,
						hostRef[index], gpuRef[index]);
				break;
			}
		}
		if (!match)
			break;
	}

	if (!match)
		printf("Arrays do not match.\n\n");
}