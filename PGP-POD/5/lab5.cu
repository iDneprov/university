#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 1024

// обработка ошибок
#define CSC(call)                                   \
do {                                                 \
	cudaError_t res = call;                             \
	if (res != cudaSuccess) {                            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",    \
			__FILE__, __LINE__, cudaGetErrorString(res));      \
		exit(0);                                              \
	}                                                        \
} while(0)

__global__ void bitonic_merge(int *host_data, int n, int mergeSize, int dist) { // разделяемая память
		int id = blockIdx.x * blockDim.x + threadIdx.x;
		int offset = blockDim.x * gridDim.x;

		int id_in_block = threadIdx.x; // индекс в блоке
		int tmp, power_of_2;
		// разделяемая память выделяется на блок, она сильно бытрее глобальной и среди потоков индексация не отличается
    __shared__ int shared_mem_block[BLOCK_SIZE];
    for (int block_part = 0; (block_part < n); block_part += offset, id += offset) { // проходим куски блоков (блок может не помещаться в сетку)
        if (block_part + blockIdx.x * blockDim.x >= n) // если мы вылезли за заданное количство чисел, то не сортируем
						break;

				__syncthreads(); // синхронизация потоков исключительно на GPU в рамках блока для работы с разделяемой памятью
        shared_mem_block[id_in_block] = host_data[block_part + blockIdx.x * blockDim.x + id_in_block];
        for (int half_cleaner_index = mergeSize; half_cleaner_index >= 2; half_cleaner_index >>= 1) { // полуочистители
            __syncthreads();
						power_of_2 = half_cleaner_index >> 1;
						if ((((id_in_block / (power_of_2)) & 1) - 1) && // нужно ли сравнивать этот элемент (выходит ли из него стрелка)
								((((id / dist) & 1) == 0 && shared_mem_block[id_in_block] > shared_mem_block[id_in_block + (power_of_2)]) || // если элемент больше и стрелка вперед
								(((id / dist) & 1) == 1 && shared_mem_block[id_in_block] < shared_mem_block[id_in_block + (power_of_2)]))) { // если элемент меньше и стрелка назад

								// свапаем элементы
								tmp = shared_mem_block[id_in_block];
                shared_mem_block[id_in_block] = shared_mem_block[id_in_block + (power_of_2)];
                shared_mem_block[id_in_block + (power_of_2)] = tmp;
        		}
		        __syncthreads();
		        host_data[block_part + blockIdx.x * blockDim.x + id_in_block] = shared_mem_block[id_in_block];
				}
    }
}

__global__ void bitonic_half_cleaner(int *host_data, int n, int half_cleaner_index, int dist) { // применить битонический полуочиститель с использованием глобальной памяти
    int id_in_block = blockIdx.x * blockDim.x + threadIdx.x;
		int offset = blockDim.x * gridDim.x;

		int tmp, power_of_2 = half_cleaner_index >> 1;
    for (int i = id_in_block; i < n; i += offset) {
        if ((((i / (power_of_2)) & 1) - 1) && // нужно ли сравнивать этот элемент (выходит ли из него стрелка)
						((((i / dist) & 1) == 0 && host_data[i] > host_data[i + (power_of_2)]) || // если элемент больше и стрелка вперед
						(((i / dist) & 1) == 1 && host_data[i] < host_data[i + (power_of_2)]))) { // если элемент меньше и стрелка назад

						// свапаем элементы
						tmp = host_data[i];
            host_data[i] = host_data[i + (power_of_2)];
            host_data[i + (power_of_2)] = tmp;
        }
    }
}

int main() {
		int n;
    fread(&n, sizeof(int), 1, stdin);

		// находим ближайшую степень двойки
    int power = 0;
    for (int i = n; i > 0; i = i >> 1)
  		++power;
  	int power_of_2 = 1 << power;

		// выделяем память для CPU
		int *host_memorry = NULL;
    host_memorry = (int *)malloc(power_of_2 * sizeof(int));
		int *host_data;
		host_data = host_memorry;

		// выделяем глобальную память для GPU
		int *device_memorry = NULL;
    CSC(cudaMalloc(&device_memorry, power_of_2 * sizeof(int)));
		int *device_data;
		device_data = device_memorry;

		// считываем числа для сортировки и дополняем оставшееся место INT_MAX
		fread(host_data, sizeof(int), n, stdin);
		for(int i = n; i < power_of_2; i++)
				host_data[i] = INT_MAX;
    cudaMemcpy(device_data, host_data, power_of_2 * sizeof(int), cudaMemcpyHostToDevice);

		// битоническая сортировка
		int half_cleaner_index;
    for (int mergeSize = 2; mergeSize <= power_of_2; mergeSize <<= 1) { // размер слияния
        if (mergeSize <= BLOCK_SIZE) { // если размер слияния меньше размера блока либо равен ему
            half_cleaner_index = mergeSize;
        } else { // в противном случае применяем битонический полуочиститель на глобальной памяти
						for (half_cleaner_index = mergeSize; half_cleaner_index > BLOCK_SIZE; half_cleaner_index >>= 1)
								bitonic_half_cleaner<<<dim3(BLOCK_SIZE / 2), dim3(BLOCK_SIZE)>>>(device_data, power_of_2, half_cleaner_index, mergeSize);
        }
				// когда становимся меньше разделяемой памяти, применяем слияние
        bitonic_merge<<<dim3(BLOCK_SIZE / 2), dim3(BLOCK_SIZE)>>>(device_data, power_of_2, half_cleaner_index, mergeSize);
    }

    cudaMemcpy(host_data, device_data, n * sizeof(int), cudaMemcpyDeviceToHost);
    fwrite(host_data, sizeof(int), n, stdout);

    free(host_memorry);
    cudaFree(device_memorry);
    return 0;
}
