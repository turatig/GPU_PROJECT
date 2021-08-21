#include <stdio.h>

int main(){
	cudaDeviceProp devProps;
	cudaGetDeviceProperties(&devProps,0);
	printf("Device 0 name: %s\n",devProps.name);
	printf("Compute capability %d.%d\n",devProps.major,devProps.minor);
}
