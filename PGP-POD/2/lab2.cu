#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// обработка ошибок
#define CSC(call)                                   \
do {                                                 \
	cudaError_t res = call;                             \
	if (res != cudaSuccess) {                            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",    \
			__FILE__, __LINE__, cudaGetErrorString(res));      \
		exit(0);                                              \
	}                                                        \
} while(0) // чтобы в блок запихнуть макрос

// проверка выхода за раницу (если вышли за границу, интерпретируем это как значение на границе)
#define xp(x) (max(min((x), w - 1), 0))
#define yp(y) (max(min((y), h - 1), 0))

#define R_MAX 1024 // максимальный радиус размытия

double coeff[2 * R_MAX + 1]; // +1 для 0
__constant__ double COEFF[2 * R_MAX + 1];

/* текстурная ссылка <тип элементов, размерность, режим нормализации>
объявлять глобально
 cudaReadModeNormalizedFloat :
 Исходный массив содержит данные в integer,
 возвращаемое значение во floating point представлении
(доступный диапазон значений отображается в интервал
[0, 1] или [-1,1])

cudaReadModeElementType
 Возвращаемое значение то же, что и во внутреннем
представлении
*/
texture<uchar4, 2, cudaReadModeElementType> x_tex;
/*
Память:
Глобальная -- медленная, пользоваться просто
Текстураная -- кешируется, есть API, обращение к ней быстрее из-за кеширование, но работать с ней менее удобно
Константная -- обращение быстрее, чем к глобальной и меньше конфликтов при одновременном чтении разными потоками за счёт дублирования между варпами (12 потоков)

Для объединения запросов нужно хранить данные без пробелов и тогда считывать мы сможем блоками а не поэлементно
*/

__global__
void calculate_f(double4 *f, int w, int h, int r)
{
		// idx - номер конкретного потока
		// blockIdx -- номер блока
		// blockDim -- размер блока
		// threadIdx -- номер потока в блоке
		// gridDim -- количество блоков
    int idx, idy, offsetx, offsety,
        x, y, u;
    double sumx, sumy, sumz;
    uchar4 pixel;
    idx = blockDim.x * blockIdx.x + threadIdx.x;
    idy = blockDim.y * blockIdx.y + threadIdx.y;
    offsetx = blockDim.x * gridDim.x;
    offsety = blockDim.y * gridDim.y;

		/*
		<<<dim3(2, 2) -- размер сетки, dim3(5, 2) -- оазмер блока>>>
		x →
		y ↓
			 (0, 0)     (1, 0)
		0 1 2 3 4 | 0  1  2 3 4
		5 6 7 8 9 | 5  6  7 8 9
		-----------------------
		0 1 2 3 4 | 0 (1) 2 3 4
		5 6 7 8 9 | 5  6  7 8 9
			(1, 0)     (1, 1)

		idx = 6 = 5 * 1 + 1
		idy = 2 = 2 * 1 + 0
		*/

    for (y = idy; y < h; y += offsety)
    {
        for (x = idx; x < w; x += offsetx)
        {
            sumx = sumy = sumz = 0;

            for (u = -r; u <= r; u++) // просчёт взвешенной суммы
            {
                pixel = tex2D(x_tex, xp(x + u), yp(y));

                sumx += COEFF[u + r] * pixel.x;
                sumy += COEFF[u + r] * pixel.y;
                sumz += COEFF[u + r] * pixel.z;
            }
            f[y * w + x] = make_double4(
								(unsigned char) (sumx),
								(unsigned char) (sumy),
								(unsigned char) (sumz), 0.0f);
        }
    }
}

__global__
void calculate_y(uchar4 *Y, const double4 *f, int w, int h, int r)
{
    int idx, idy, offsetx, offsety,
        x, y, v;
    double sumx, sumy, sumz;
    double4 pixel;
    idx = blockDim.x * blockIdx.x + threadIdx.x;
    idy = blockDim.y * blockIdx.y + threadIdx.y;
    offsetx = blockDim.x * gridDim.x;
    offsety = blockDim.y * gridDim.y;

    for (y = idy; y < h; y += offsety)
    {
        for (x = idx; x < w; x += offsetx)
        {
            sumx = sumy = sumz = 0;

            for (v = -r; v <= r; v++) // просчёт взвешенной суммы
            {
                pixel = f[yp(y + v) * w + xp(x)];

                sumx += COEFF[v + r] * pixel.x;
                sumy += COEFF[v + r] * pixel.y;
                sumz += COEFF[v + r] * pixel.z;
            }
            Y[y * w + x] = make_uchar4(
								(unsigned char) (sumx),
								(unsigned char) (sumy),
								(unsigned char) (sumz), 0.0f);
        }
    }
}

int main()
{
    int w, h,
        r,
        u;
    void *MEM_CPU = NULL, *MEM_GPU = NULL;
    uchar4 *im,
           *y;
    double norm;
    double4 *f;
    cudaArray *x = NULL;
    cudaChannelFormatDesc ch_x;

    FILE *fp;
    char name_src_im[256], name_dst_im[256];

    dim3 blocks2D  = dim3(255, 255),
         threads2D = dim3(32, 8);

    scanf("%s\n%s\n%d", name_src_im, name_dst_im, &r);
    fp = fopen(name_src_im, "rb");
    if (fp == NULL)
    {
        fprintf(stderr, "Error: can't open %s\n", name_src_im);
        return 0;
    }
    fread(&w, sizeof(int), 1, fp);
    fread(&h, sizeof(int), 1, fp);

    /* Выделение памяти на CPU */

    MEM_CPU = malloc(w * h * sizeof(uchar4));
    if (MEM_CPU == NULL)
    {
        fprintf(stderr, "Error: not enough memory in CPU\n");
        free(MEM_CPU);
        return 0;
    }
    im = (uchar4 *)MEM_CPU;

    /* Конец выделения памяти на CPU */

    /* Выделение памяти на GPU */

    CSC(cudaMalloc(&MEM_GPU, w * h * sizeof(uchar4) + w * h * sizeof(double4))); // память под f и y одним malloc
    f = (double4 *) MEM_GPU;
    y = (uchar4 *)(f + w * h);

    ch_x = cudaCreateChannelDesc<uchar4>(); // дескриптор (размер + форма + тип)
    CSC(cudaMallocArray(&x, &ch_x, w, h)); // выделяем память для двумерного массива

    /* Конец выделения памяти на GPU */

    fread(im, sizeof(uchar4), w * h, fp);
    fclose(fp);

    cudaMemcpyToArray(x, 0, 0, im, w * h * sizeof(uchar4), cudaMemcpyHostToDevice); // 0, 0 -- смещение (нужно для обрезки)

    /* Подготовка текстурной ссылки, настройка интерфейса работы с данными */

    // Политика обработки выхода за границы по каждому измерению
    x_tex.addressMode[0] = cudaAddressModeClamp; // обработка выхода за границы по Х
    x_tex.addressMode[1] = cudaAddressModeClamp; // cudaAddressModeClamp делает то же самое, что и #define yp(y) (max(min((y), h - 1), 0)) поэтому эти макросы бесполезны
    x_tex.channelDesc = ch_x; // дескриптор
    x_tex.filterMode = cudaFilterModePoint; // Без интерполяции при обращении по дробным координатам
    x_tex.normalized = false; // Режим нормализации координат: без нормализации можно нормолизовать и образаться по дробным координатам от -1 или 0 до 1

		/*
		cudaAddressModeClamp

		The signal c[k] is continued outside k=0,...,M-1 so that c[k] = c[0] for k < 0, and c[k] = c[M-1] for k >= M.

		cudaAddressModeBorder

		The signal c[k] is continued outside k=0,...,M-1 so that c[k] = 0 for k < 0and for k >= M.

		Now, to describe the last two address modes, we are forced to consider normalized coordinates, so that the 1D input signal samples are assumed to be c[k / M], with k=0,...,M-1.

		cudaAddressModeWrap

		The signal c[k / M] is continued outside k=0,...,M-1 so that it is periodic with period equal to M. In other words, c[(k + p * M) / M] = c[k / M] for any (positive, negative or vanishing) integer p.

		cudaAddressModeMirror

		The signal c[k / M] is continued outside k=0,...,M-1 so that it is periodic with period equal to 2 * M - 2. In other words, c[l / M] = c[k / M] for any l and k such that (l + k)mod(2 * M - 2) = 0.
		*/

    // Связываем интерфейс с данными
    cudaBindTextureToArray(x_tex, x, ch_x);

    if (r == 0) // вырожденный случай
    {
        cudaMemcpy(y, im, w * h * sizeof(uchar4), cudaMemcpyHostToDevice);
    }
    else
    {
				// заранее вычисляем коэффициенты и закидываем их в константную память
        norm = 0;
        for (u = -r; u <= r; u++)
        {
            norm += exp(-(u * u) / (2. * r * r));
        }
        for (u = -r; u <= r; u++)
        {
            coeff[u + r] = exp(-(u * u) / (2. * r * r)) / norm;
        }
        CSC(cudaMemcpyToSymbol(COEFF, coeff, (2 * r + 1) * sizeof(double), 0, cudaMemcpyHostToDevice)); // копирование в константную память 0 -- смещение

        calculate_f<<<blocks2D, threads2D>>>(f, w, h, r);
        CSC(cudaGetLastError());

        calculate_y<<<blocks2D, threads2D>>>(y, f, w, h, r);
        CSC(cudaGetLastError());
    }
    CSC(cudaMemcpy(im, y, w * h * sizeof(uchar4), cudaMemcpyDeviceToHost));

    fp = fopen(name_dst_im, "wb");
    if (fp == NULL)
    {
        fprintf(stderr, "Error: can't open %s\n", name_dst_im);
        free(MEM_CPU);
        cudaFree(MEM_GPU);
        cudaFreeArray(x);
    }
    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(im, sizeof(uchar4), w * h, fp);
    fclose(fp);

    cudaUnbindTexture(&x_tex);
    free(MEM_CPU);
    cudaFree(MEM_GPU);
    cudaFreeArray(x);
    return 0;
}
