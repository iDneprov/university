#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CSC(call)                                   \
do {                                                 \
	cudaError_t res = call;                             \
	if (res != cudaSuccess) {                            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",    \
			__FILE__, __LINE__, cudaGetErrorString(res));      \
		exit(0);                                              \
	}                                                        \
} while(0)

struct abs_comparator
{
    __host__ __device__ bool operator()(double a, double b)
    {
        return fabs(a) < fabs(b);
    }
};

__global__
void swap_rows(double *A, int m, int n, int i1, int i2)
{
    int i,
        id = blockIdx.x * blockDim.x + threadIdx.x,
        offset = blockDim.x * gridDim.x;
    double t;

    for (i = id; i < n; i += offset)
    {
        t = A[i * m + i1];
        A[i * m + i1] = A[i * m + i2];
        A[i * m + i2] = t;
    }
}

__global__
void direct_move_kernel(double *A, int m, int n, int lead_i, int lead_j)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x,
        idy = blockIdx.y * blockDim.y + threadIdx.y,
        offsetx = blockDim.x * gridDim.x,
        offsety = blockDim.y * gridDim.y,
        i, j;

    for (j = idy; j < n - lead_j - 1; j += offsety) // цикл по столбцам
    {
        for (i = idx; i < m - lead_i - 1; i += offsetx) // цикл по строкам
        {
            A[(lead_j + 1) * m + j * m + lead_i + 1 + i] -= A[(lead_j + 1) * m + j * m + lead_i] *
                                                            A[lead_j * m + lead_i + 1 + i] /
                                                            A[lead_j * m + lead_i];
        }
    }
}

__global__
void backward_move_kernel(const double *A, int n, double *b, int lead_i, int lead_j)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x,
        offsetx = blockDim.x * gridDim.x,
        i;

    for (i = idx; i < lead_i; i += offsetx) // цикл по строкам
        b[i] -= b[lead_i] * A[lead_j * n + i] / A[lead_j * n + lead_i];
}

__global__
void division_kernel(const double *A, int n, double *b)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x,
        offsetx = blockDim.x * gridDim.x,
        i;

    for (i = idx; i < n; i += offsetx) // цикл по строкам
        b[i] /= A[i * n + i];

}

int main()
{
    int n,
        i, j,
        buf_size,
        lead_elem_ind;

    void *MEM_CPU = NULL, *MEM_GPU = NULL;
    double *buf,
           *A, *b;

    dim3 blocks2D =  dim3(512, 256),
         threads2D = dim3(32, 16),
         blocks1D =  dim3(512),
         threads1D = dim3(512);

    thrust::device_ptr<double> start_elem, lead_elem;
    abs_comparator comp;

    scanf("%d", &n);
    buf_size = n * n;

    MEM_CPU = malloc(buf_size * sizeof(double));
    if (MEM_CPU == NULL)
    {
        fprintf(stderr, "ERROR: Not enough Memory on CPU\n");
        return 0;
    }
    buf = (double *)MEM_CPU;

    CSC(cudaMalloc(&MEM_GPU, (n * n + n) * sizeof(double)));

    A = (double *)MEM_GPU;
    b = A + n * n;

    // Чтение A
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            scanf("%lf", buf + j * n + i);
        }
    }

    cudaMemcpy(A, buf, n * n * sizeof(double), cudaMemcpyHostToDevice);

    // Чтение b
    for (i = 0; i < n; i++)
    {
        scanf("%lf", buf + i);
    }

    cudaMemcpy(b, buf, n * sizeof(double), cudaMemcpyHostToDevice);

    // прямой ход метода Гаусса (получаем триугольную матрицу)
    for(j = 0; j < n; j++) // цикл по столбцам
    {
        start_elem = thrust::device_pointer_cast(A + j * n + j);
        lead_elem = thrust::max_element(start_elem, start_elem + n - j, comp);
        lead_elem_ind = (int)(lead_elem - start_elem) + j;

        if (j != lead_elem_ind) // если ведущий элемент в другом столбце, меняем столбцы местами
            swap_rows<<<blocks1D, threads1D>>>(A + j * n, n, n + 1 - j, j, lead_elem_ind);

        direct_move_kernel<<<blocks2D, threads2D>>>(A, n, n + 1, j, j);

        CSC(cudaGetLastError());
    }

    // обратный ход метода Гаусса (получаем диагональную матрицу)
    for (j = n - 1; j > 0; j--) // цикл по ступенькам
    {
        backward_move_kernel<<<blocks1D, threads1D>>>(A, n, b, j, j);
        CSC(cudaGetLastError());
    }

		// получаем единичную матрицу
    division_kernel<<<blocks1D, threads1D>>>(A, n, b);
    CSC(cudaGetLastError());

    cudaMemcpy(buf, b, n * sizeof(double), cudaMemcpyDeviceToHost);

		// выводим результат
    for (i = 0; i < n; i++)
        printf("%.10le ", buf[i]);
    printf("\n");

    free(MEM_CPU);
    cudaFree(MEM_GPU);
    return 0;
}
