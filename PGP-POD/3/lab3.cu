#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <vector>

#define CSC(call)                                   \
do {                                                 \
	cudaError_t res = call;                             \
	if (res != cudaSuccess) {                            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",    \
			__FILE__, __LINE__, cudaGetErrorString(res));      \
		exit(0);                                              \
	}                                                        \
} while(0)

typedef unsigned char uchar;

#define NC_MAX 32 // максимальное количество классов

double3 avg[NC_MAX]; // средние
double cov[3 * 3 * NC_MAX]; // ковариационные матрицы
__constant__ double3 AVG[NC_MAX];
__constant__ double COV[3 * 3 * NC_MAX];

__host__ __device__
double dot(const double *A, const double *a, const double *b, int n)
{
    /*
    * res = a ^ T * A * b
    */
    int i, j;
    double res = 0;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            res += A[i * n + j] * a[i] * b[j];
        }
    }
    return res;
}

__host__ __device__ // работает везде
void mat_mul(const double *A, const double *B, double *C, int m, int n, int l, double alpha=1, double beta=0)
{
    /*
    * C = alpha * A * B + beta * C
    *
    * m - число строк A
    * n - число столбцов A и число строк B
    * l - число столбцов B
    */
    int i, j, k;
    double dot;

    for (i = 0; i < m; i++)
    {
        for (j = 0; j < l; j++)
        {
            dot = 0;
            for (k = 0; k < n; k++)
            {
                dot += A[i * n + k] * B[k * l + j];
            }
            C[i * l + j] = alpha * dot + (beta == 0 ? 0 : beta * C[i * l + j]);
        }
    }
}

__host__ __device__
void mat_mul_C(const double *A, double *B, double c, int m, int n)
{
    /*
    * B = c * A
    */
    int i, j;
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            B[i * n + j] = c * A[i * n + j];
        }
    }
}

__host__ __device__
void mat_sum(const double *A, const double *B, double *C, int m, int n, double alpha=1, double beta=1)
{
    /*
    * C = alpha * A + beta * B
    */
    int i, j;
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            C[i * n + j] = alpha * A[i * n + j] + beta * B[i * n + j];
        }
    }
}

__host__ __device__
void mat_set(double *A, int n, double alpha=0, double beta=0)
{
    /*
    * Инициализация A значениями alpha на диагонали и
    * значениями beta вне диагонали
    */
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            if (i == j)
            {
                A[i * n + j] = alpha;
            }
            else
            {
                A[i * n + j] = beta;
            }
        }
    }
}

void mat_swap_rows(double *A, int n, int i1, int i2)
{
    int j;
    double tmp;
    for (j = 0; j < n; j++)
    {
        tmp = A[i1 * n + j];
        A[i1 * n + j] = A[i2 * n + j];
        A[i2 * n + j] = tmp;
    }
}

void mat_transpose(const double *A, double *A_t, int n)
{
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            A_t[i * n + j] = A[j * n + i];
        }
    }
}

bool LUP(double *A, int n, int *pi, double eps=1e-10)
{
    int i, j, k, k_;
    int tmp;

    for (i = 0; i < n; i++)
    {
        pi[i] = i;
    }

    for (k = 0; k < n; k++)
    {
        k_ = k;
        for (i = k + 1; i < n; i++)
        {
            if (fabs(A[i * n + k]) > fabs(A[k_ * n + k]))
            {
                k_ = i;
            }
        }

        if (fabs(A[k_ * n + k]) < eps)
        {
            return false;
        }

        if (k != k_)
        {
            tmp = pi[k];
            pi[k] = pi[k_];
            pi[k_] = tmp;
            mat_swap_rows(A, n, k, k_);
        }

        for (i = k + 1; i < n; i++)
        {
            A[i * n + k] /= A[k * n + k];
            for (j = k + 1; j < n; j++)
            {
                A[i * n + j] -= A[i * n + k] * A[k * n + j];
            }
        }
    }
    return true;
}

void LUP_solve(const double *LU, const int *pi, const double *b, int n, double *x, double *work)
{
    /*
    * work - вектор размерности n
    */
    int i, j;
    double sum;
    double *y = work;

    memset(y, 0, n * sizeof(double));

    for (i = 0; i < n; i++) // прямой ход
    {
        for (sum = 0, j = 0; j <= i - 1; sum += LU[i * n + j] * y[j], j++);
        y[i] = b[pi[i]] - sum;
    }

    for (i = n - 1; i >= 0; i--) // обратный ход
    {
        for (sum = 0, j = i + 1; j < n; sum += LU[i * n + j] * x[j], j++);
        x[i] = (y[i] - sum) / LU[i * n + i];
    }
}

void LUP_inv_mat(double *A, double *A_inv, int n, double *work, int *iwork, double eps=1e-10) // инвариантная матрица при помощи LUP разложения (украдено из Кормана)
{
    /*
    * work  - вектор длины n ^ 2 + 2 * n
    * iwork - вектор длины n
    */
    int i;
    double *X, *e, *space;
    int *pi;

    X = work;
    e = X + n * n;
    space = e + n;

    pi = iwork;

    memset(e, 0, n * sizeof(double));
    e[0] = 1;

    LUP(A, n, pi, eps);
    for (i = 0; i < n - 1; i++)
    {
        LUP_solve(A, pi, e, n, X + i * n, space);
        e[i] = 0;
        e[i + 1] = 1;
    }
    LUP_solve(A, pi, e, n, X + i * n, space);
    mat_transpose(X, A_inv, n);
}

__global__
void kernel(int nc, uchar4 *im, int w, int h)
{
    int i, j, jc, idx;
    int offset;
    double3 to_dot;
    double dist, dist_max;

    idx = blockDim.x * blockIdx.x + threadIdx.x; // абсолютный номер потока
    offset = blockDim.x * gridDim.x; // общее кол-во потоков

    for (i = idx; i < w * h; i += offset) // идем по всем пикселям
    {
        dist_max = -INFINITY;
        jc = 0;
        for (j = 0; j < nc; j++) // цикл по числу классов
        {

            to_dot = make_double3((double)im[i].x - AVG[j].x,
								(double)im[i].y - AVG[j].y,
								(double)im[i].z - AVG[j].z);
            dist = -dot(COV + j * 3 * 3, (double *)&to_dot, (double *)&to_dot, 3); // произведение из основной формулы
            if (dist > dist_max) // для каждого класса вычисляем dist и выбираем максимум
            {
                jc = j;
                dist_max = dist;
            }
        }
        im[i].w = jc; // записываем номер класса с max dist
    }
}

int main()
{
    int nc, np, i, j, x, y, w, h;
    uchar4 *im = NULL, *im_dev = NULL;
    std::vector<double3> v(0);
    double mat[3 * 3],
           work[3 * 3 + 2 * 3];
    int iwork[3];

    FILE *fp;
    char name_src_im[256], name_dst_im[256];

    dim3 blocks(256), threads(256);

    scanf("%s\n%s\n%d", name_src_im, name_dst_im, &nc);

    fp = fopen(name_src_im, "rb");
    if (fp == NULL)
    {
        fprintf(stderr, "Error: can't open %s\n", name_src_im);
        return 0;
    }
    fread(&w, sizeof(int), 1, fp);
    fread(&h, sizeof(int), 1, fp);

    im = (uchar4 *)malloc(w * h * sizeof(uchar4));
    if (im == NULL)
    {
        fprintf(stderr, "Error: not enough memory in CPU\n");
        goto FREE;
    }
    CSC(cudaMalloc(&im_dev, w * h * sizeof(uchar4)));

    fread(im, sizeof(uchar4), w * h, fp); // считывание пикселей
    fclose(fp);

    for (j = 0; j < nc; j++) // цикл по числу классов
    {
        scanf("%d", &np); // количество пикселей в классе
        if(v.size() < np) v.resize(np);

        avg[j] = make_double3(0, 0, 0);
        for (i = 0; i < np; i++) // просчёт среднего по j-му классу
        {
            scanf("%d %d", &x, &y);
            v[i] = make_double3((double)im[y * w + x].x,
								(double)im[y * w + x].y,
								(double)im[y * w + x].z);

            avg[j].x += v[i].x;
            avg[j].y += v[i].y;
            avg[j].z += v[i].z;
        }
        avg[j].x /= np;
        avg[j].y /= np;
        avg[j].z /= np;

        if (np > 1)
        {
            mat_set(mat, 3, 0, 0);          // инициализация нулями
            for (i = 0; i < np; i++)        // просчёт ковариации по i-му классу
            {
                v[i].x = v[i].x - avg[j].x;
                v[i].y = v[i].y - avg[j].y;
                v[i].z = v[i].z - avg[j].z;
								// double3 приводим к вектору double
                mat_mul((double *)(v.data() + i), (double *)(v.data() + i), mat, 3, 1, 3, 1, 1); // умножение векторов как матриц 3х1 1х3 получаем 3х3
            }
            mat_mul_C(mat, mat, 1. / (np - 1), 3, 3); // делим матрицу на коэффициент
            LUP_inv_mat(mat, cov + j * 3 * 3, 3, work, iwork); // получаем матрицу ковариации для j-того класса (work, iwork -- место, чтобы внутри функции не делать malloc)
        }
        else
        {
            mat_set(cov + j * 3 * 3, 3, 1, 0); // единичная матрица
        }
    }

    /* Копирование в константную память */
    CSC(cudaMemcpyToSymbol(AVG, avg, nc * sizeof(double3), 0, cudaMemcpyHostToDevice));
    CSC(cudaMemcpyToSymbol(COV, cov, nc * 3 * 3 * sizeof(double), 0, cudaMemcpyHostToDevice));

    /* Копирование изображения */
    cudaMemcpy(im_dev, im, w * h * sizeof(uchar4), cudaMemcpyHostToDevice); // копирование изображения

    kernel<<<blocks, threads>>>(nc, im_dev, w, h);
    CSC(cudaGetLastError());
    CSC(cudaMemcpy(im, im_dev, w * h * sizeof(uchar4), cudaMemcpyDeviceToHost));

    fp = fopen(name_dst_im, "wb");
    if (fp == NULL)
    {
        fprintf(stderr, "Error: can't open %s\n", name_dst_im);
        goto FREE;
    }
    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(im, sizeof(uchar4), w * h, fp);
    fclose(fp);

FREE:
    free(im);
    cudaFree(im_dev);
    return 0;
}
