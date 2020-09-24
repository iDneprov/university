#include <iostream>
#include <iomanip>

__global__ void KernelVectorAbs(double *vector, int size){
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int offset = gridDim.x * blockDim.x;
  while (i < size) {
    vector[i] = (vector[i] < 0) ? -vector[i] : vector[i];
    i += offset;
  }
}

int main(){
  std::ios_base::sync_with_stdio(false);
  int size, blocks, threds;
  std::cin >> blocks >> threds >> size;
  double *vector = new double[size], *result = new double[size], *cudaVecor;
  for (int i = 0; i < size; ++i)
    std::cin >> vector[i];

  cudaMalloc((void **) &cudaVecor, sizeof(double) * size);
  cudaMemcpy(cudaVecor, vector, sizeof(double) * size, cudaMemcpyHostToDevice);
  KernelVectorAbs<<<blocks, threds>>>(cudaVecor, size);
  cudaGetLastError();
  cudaMemcpy(result, cudaVecor, sizeof(double) * size, cudaMemcpyDeviceToHost);

		std::cout.setf(std::ios::scientific);
  std::cout.precision(10);
		for (int i = 0; i < size; ++i)
    std::cout << result[i] << ' ';
  std::cout << '\n';

  cudaFree(cudaVecor);
  delete[] vector;
}
