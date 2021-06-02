#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>

#include "mpi.h"

// индексация внутри блока
#define xyz2i(x, y, z) (((z) + 1) * (bx + 2) * (by + 2) + ((y) + 1) * (bx + 2) + (x) + 1)

// индексация по процессам
#define ijk2i(i, j, k) ((k) * px * py + (j) * px + (i))
#define i2k(i) ((i) / (px * py))
#define i2j(i) (((i) - i2k(i) * px * py) / px)
#define i2i(i) (((i) - i2k(i) * px * py) % px)

#define max(x, y) ((x) > (y) ? (x) : (y))

#define is_left (i2i(id) == 0)
#define is_right (i2i(id) == px - 1)
#define is_front (i2j(id) == 0)
#define is_back (i2j(id) == py - 1)
#define is_down (i2k(id) == 0)
#define is_up (i2k(id) == pz - 1)

#define N_CHARS_IN_DOUBLE 14

#define vector_print(vec, n, format)            \
{                                               \
    int i;                                      \
    for (i = 0; i < (n); i++)                   \
    {                                           \
        printf(#format" ", (vec)[i]);           \
    }                                           \
    printf("\n");                               \
}

#define vector_print_64F(vec, n) vector_print(vec, n, %7le)
#define vector_print_32S(vec, n) vector_print(vec, n, %d)

#define CSC(call)                                                   \
do                                                                  \
{                                                                   \
    cudaError_t res = call;                                         \
    if (res != cudaSuccess)                                         \
    {                                                               \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(res));       \
        fflush(stderr);                                             \
        cudaFree(MEM_GPU);                                          \
        free(MEM_CPU);                                              \
        return 0;                                                   \
    }                                                               \
} while(0)

// Шаг алгоритма
__global__ void kernel(const double *data_prev, double *data,
                       int bx, int by, int bz, 
                       double hx, double hy, double hz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x,
        idy = blockIdx.y * blockDim.y + threadIdx.y,
        idz = blockIdx.z * blockDim.z + threadIdx.z,
        offsetx = blockDim.x * gridDim.x,
        offsety = blockDim.y * gridDim.y,
        offsetz = blockDim.z * gridDim.z,
        x, y, z; 
    for(z = idz; z < bz; z += offsetz)
        for(y = idy; y < by; y += offsety)
            for(x = idx; x < bx; x += offsetx)
                data[xyz2i(x, y, z)] = ((data_prev[xyz2i(x + 1, y, z)] + data_prev[xyz2i(x - 1, y, z)]) / (hx * hx) +
                                        (data_prev[xyz2i(x, y + 1, z)] + data_prev[xyz2i(x, y - 1, z)]) / (hy * hy) +
                                        (data_prev[xyz2i(x, y, z + 1)] + data_prev[xyz2i(x, y, z - 1)]) / (hz * hz)) /
                                       (2 / (hx * hx) + 2 / (hy * hy) + 2 / (hz * hz));
}

// Инициализация вектора значение val
__global__ void vector_set(double *dst, int n,
                           double val)
{
   int id = blockIdx.x * blockDim.x + threadIdx.x,
       offset = blockDim.x * gridDim.x,
       i;
       
    for(i = id; i < n; i += offset)
        dst[i] = val;
}

// Создание тензора погрешностей
__global__ void data_diff_and_fabs(const double *a, const double *b, double *c,
                                   int bx, int by, int bz)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x,
       idy = blockIdx.y * blockDim.y + threadIdx.y,
       idz = blockIdx.z * blockDim.z + threadIdx.z,
       offsetx = blockDim.x * gridDim.x,
       offsety = blockDim.y * gridDim.y,
       offsetz = blockDim.z * gridDim.z,
       x, y, z;
    for(z = idz; z < bz; z += offsetz)
        for(y = idy; y < by; y += offsety)
            for(x = idx; x < bx; x += offsetx)
                c[z * bx * by + y * bx + x] = fabs(a[xyz2i(x, y, z)] - b[xyz2i(x, y, z)]);
}

// Инициализация матрицы dst значением val со сдвигами
__global__ void mat_set(double *dst, int stepdst, int lddst,
                        int n, int m,
                        double val)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x,
       idy = blockIdx.y * blockDim.y + threadIdx.y,
       offsetx = blockDim.x * gridDim.x,
       offsety = blockDim.y * gridDim.y,
       x, y;
       
    for(y = idy; y < m; y += offsety)
        for(x = idx; x < n; x += offsetx)
        {
            dst[y * lddst + x * stepdst] = val;
        }
}

// Копирование матрицы src в матрицу dst со сдвигами
__global__ void mat_copy(const double *src, int stepsrc, int ldsrc,
                              double *dst, int stepdst, int lddst,
                         int n, int m)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x,
       idy = blockIdx.y * blockDim.y + threadIdx.y,
       offsetx = blockDim.x * gridDim.x,
       offsety = blockDim.y * gridDim.y,
       x, y;
       
    for(y = idy; y < m; y += offsety)
        for(x = idx; x < n; x += offsetx)
        {
            dst[y * lddst + x * stepdst] = src[y * ldsrc + x * stepsrc];
        }
}

int main(int argc, char *argv[])
{
    int id, root = 0, numproc,
        i,
        x, y, z,
        device_count,
        buf_size, block_size,
        px, py, pz,                 // размер сетки процессов
        bx, by, bz;                 // размер блока на один процесс

    void *MEM_CPU = NULL, *MEM_GPU = NULL;
    
    double *data, *data_prev, *data_tmp, *data_reduce, *buf_dev,
           *data_host, 
           *buf_left, *buf_right, *buf_front, *buf_back, *buf_up, *buf_down;
    char *buf_char;
    thrust::device_ptr<double> data_ptr;
    double eps,
           dx, dy, dz,              // размер области
           ud, uu, ul, ur, uf, ub,  // границы
           u0,                      // начальное распределение
           hx, hy, hz,              // шаги
           max_u, max_global;       // максимальные значения

    int *blocklens;
    MPI_Aint *indicies;
    MPI_File fp;
    MPI_Datatype filetype;
    
    dim3 blocks3D  = dim3(8, 8, 8),
         threads3D = dim3(8, 8, 8),
         blocks2D  = dim3(256, 256),
         threads2D = dim3(32, 8),
         blocks1D  = dim3(256),
         threads1D = dim3(256);

    char name_out[128];
    MPI_Status status;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &numproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    
    cudaGetDeviceCount(&device_count);  // каждый процесс использует
    cudaSetDevice(id % device_count);   // одну из доступных видеокарт

    if (id == root)
    {
        scanf("%d %d %d\n", &px, &py, &pz);
        scanf("%d %d %d\n", &bx, &by, &bz);
        scanf("%s\n", name_out);
        scanf("%le\n", &eps);
        scanf("%lf %lf %lf\n", &dx, &dy, &dz);
        scanf("%lf %lf %lf %lf %lf %lf\n", 
              &ud, &uu, &ul, &ur, &uf, &ub);
        scanf("%lf", &u0);

        hx = dx / (px * bx);
        hy = dy / (py * by);
        hz = dz / (pz * bz);
    }

    MPI_Bcast(&bx, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Bcast(&by, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Bcast(&bz, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Bcast(&px, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Bcast(&py, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Bcast(&pz, 1, MPI_INT, root, MPI_COMM_WORLD);

    MPI_Bcast(&ul, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);
    MPI_Bcast(&ur, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);
    MPI_Bcast(&uf, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);
    MPI_Bcast(&ub, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);
    MPI_Bcast(&ud, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);
    MPI_Bcast(&uu, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);

    MPI_Bcast(&u0, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);

    MPI_Bcast(&hx, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);
    MPI_Bcast(&hy, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);
    MPI_Bcast(&hz, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);

    MPI_Bcast(&eps, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);

    MPI_Bcast(name_out, 128, MPI_CHAR, root, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    // размер блока с виртуальными ячейками
    block_size = (bx + 2) * (by + 2) * (bz + 2);               
    MEM_CPU = malloc((block_size + 2 * (by * bx + bz * by + bz * bx)) * sizeof(double) + 
                     by * bz * sizeof(int) +
                     by * bz * sizeof(MPI_Aint) +
                     bx * by * bz * N_CHARS_IN_DOUBLE * sizeof(char));
    if(MEM_CPU == NULL)
    {
        fprintf(stderr, "Error: Not Enough memory on CPU%d\n", id);
        fflush(stderr);
        return 0;
    }
    
    data_host = (double *)MEM_CPU;
    buf_left = data_host + block_size;
    buf_right = buf_left + bz * by;
    buf_front = buf_right + bz * by;
    buf_back = buf_front + bz * bx;
    buf_up = buf_back + bz * bx;
    buf_down = buf_up + by * bx;
    blocklens = (int *)(buf_down + by * bx);
    indicies = (MPI_Aint *)(blocklens + by * bz);
    buf_char = (char *)(indicies + by * bz);

    // размер буфера на видеокарте
    buf_size = max(max(bx * by, by * bz), max(bx * bz, px * py * pz));
    CSC(cudaMalloc(&MEM_GPU, (2 * block_size + bx * by * bz + buf_size) * sizeof(double)));
    data = (double *)MEM_GPU;
    data_prev = data + block_size;
    data_reduce = data_prev + block_size;
    buf_dev = data_reduce + bx * by * bz;

    vector_set<<<blocks1D, threads1D>>>(data, block_size, u0);
    vector_set<<<blocks1D, threads1D>>>(data_prev, block_size, u0);

    if (is_left) // left
    {
        mat_set<<<blocks2D, threads2D>>>(data      + xyz2i(-1, 0, 0), bx + 2, (bx + 2) * (by + 2), 
                                         by, bz,
                                         ul);
        mat_set<<<blocks2D, threads2D>>>(data_prev + xyz2i(-1, 0, 0), bx + 2, (bx + 2) * (by + 2), 
                                         by, bz,
                                         ul);
    }
    if (is_right) // right
    {   
        mat_set<<<blocks2D, threads2D>>>(data      + xyz2i(bx, 0, 0), bx + 2, (bx + 2) * (by + 2), 
                                         by, bz, 
                                         ur);
        mat_set<<<blocks2D, threads2D>>>(data_prev + xyz2i(bx, 0, 0), bx + 2, (bx + 2) * (by + 2), 
                                         by, bz, 
                                         ur);
    }
    if (is_front) // front
    {   
        mat_set<<<blocks2D, threads2D>>>(data      + xyz2i(0, -1, 0), 1, (bx + 2) * (by + 2), 
                                         bx, bz,
                                         uf);
        mat_set<<<blocks2D, threads2D>>>(data_prev + xyz2i(0, -1, 0), 1, (bx + 2) * (by + 2), 
                                         bx, bz,
                                         uf);
    }
    if (is_back) // back
    {
        mat_set<<<blocks2D, threads2D>>>(data      + xyz2i(0, by, 0), 1, (bx + 2) * (by + 2), 
                                         bx, bz, 
                                         ub);
        mat_set<<<blocks2D, threads2D>>>(data_prev + xyz2i(0, by, 0), 1, (bx + 2) * (by + 2), 
                                         bx, bz,
                                         ub);
    }
    if (is_down) // down
    {
        mat_set<<<blocks2D, threads2D>>>(data      + xyz2i(0, 0, -1), 1, bx + 2, 
                                         bx, by, 
                                         ud);
        mat_set<<<blocks2D, threads2D>>>(data_prev + xyz2i(0, 0, -1), 1, bx + 2, 
                                         bx, by, 
                                         ud);
    }
    if (is_up) // up
    {
        mat_set<<<blocks2D, threads2D>>>(data      + xyz2i(0, 0, bz), 1, bx + 2, 
                                         bx, by,
                                         uu);
        mat_set<<<blocks2D, threads2D>>>(data_prev + xyz2i(0, 0, bz), 1, bx + 2, 
                                         bx, by, 
                                         uu);
    }

    do // основной цикл
    {
        /* обмен граничными условиями: отправка данных */

        if (!is_left)
        {
            /* left -> bufer */
            mat_copy<<<blocks2D, threads2D>>>(data + xyz2i(0, 0, 0), bx + 2, (bx + 2) * (by + 2),
                                              buf_dev, 1, by,
                                              by, bz);
                                             
            cudaMemcpy(buf_left, buf_dev, by * bz * sizeof(double), cudaMemcpyDeviceToHost);

            MPI_Send(buf_left, by * bz, MPI_DOUBLE, ijk2i(i2i(id) - 1, i2j(id), i2k(id)), 1, MPI_COMM_WORLD);
        }
        
        if (!is_front)
        {
            /* front -> bufer */
            mat_copy<<<blocks2D, threads2D>>>(data + xyz2i(0, 0, 0), 1, (bx + 2) * (by + 2),
                                              buf_dev, 1, bx,
                                              bx, bz);
                                             
            cudaMemcpy(buf_front, buf_dev, bx * bz * sizeof(double), cudaMemcpyDeviceToHost);

            MPI_Send(buf_front, bx * bz, MPI_DOUBLE, ijk2i(i2i(id), i2j(id) - 1, i2k(id)), 1, MPI_COMM_WORLD);
        }
        
        if (!is_down)
        {
            /* down -> bufer */
            mat_copy<<<blocks2D, threads2D>>>(data + xyz2i(0, 0, 0), 1, bx + 2,
                                              buf_dev, 1, bx,
                                              bx, by);
                                             
            cudaMemcpy(buf_down, buf_dev, bx * by * sizeof(double), cudaMemcpyDeviceToHost);

            MPI_Send(buf_down, bx * by, MPI_DOUBLE, ijk2i(i2i(id), i2j(id), i2k(id) - 1), 1, MPI_COMM_WORLD);
        }
        
        //==============================================================
        
        if (!is_right)
        {
            MPI_Recv(buf_right, by * bz, MPI_DOUBLE, ijk2i(i2i(id) + 1, i2j(id), i2k(id)), 1, MPI_COMM_WORLD, &status);

            /* right update */
            cudaMemcpy(buf_dev, buf_right, by * bz * sizeof(double), cudaMemcpyHostToDevice);
            
            mat_copy<<<blocks2D, threads2D>>>(buf_dev, 1, by,
                                              data + xyz2i(bx, 0, 0), bx + 2, (bx + 2) * (by + 2),
                                              by, bz);
        }
        
        if (!is_back)
        {
            MPI_Recv(buf_back, bx * bz, MPI_DOUBLE, ijk2i(i2i(id), i2j(id) + 1, i2k(id)), 1, MPI_COMM_WORLD, &status);

            /* back update */
            cudaMemcpy(buf_dev, buf_back, bx * bz * sizeof(double), cudaMemcpyHostToDevice);

            mat_copy<<<blocks2D, threads2D>>>(buf_dev, 1, bx,
                                              data + xyz2i(0, by, 0), 1, (bx + 2) * (by + 2),
                                              bx, bz);
        }
        
        if (!is_up)
        {
            MPI_Recv(buf_up, bx * by, MPI_DOUBLE, ijk2i(i2i(id), i2j(id), i2k(id) + 1), 1, MPI_COMM_WORLD, &status);

            /* up update */
            cudaMemcpy(buf_dev, buf_up, bx * by * sizeof(double), cudaMemcpyHostToDevice);
            
            mat_copy<<<blocks2D, threads2D>>>(buf_dev, 1, bx,
                                              data + xyz2i(0, 0, bz), 1, bx + 2,
                                              bx, by);
        }
        
        //==============================================================
        
        if (!is_right)
        {
            /* right -> bufer */
            mat_copy<<<blocks2D, threads2D>>>(data + xyz2i(bx - 1, 0, 0), bx + 2, (bx + 2) * (by + 2),
                                              buf_dev, 1, by,
                                              by, bz);
                                             
            cudaMemcpy(buf_right, buf_dev, by * bz * sizeof(double), cudaMemcpyDeviceToHost);

            MPI_Send(buf_right, by * bz, MPI_DOUBLE, ijk2i(i2i(id) + 1, i2j(id), i2k(id)), 1, MPI_COMM_WORLD);
        }
        
        if (!is_back)
        {
            /* back -> bufer */
            mat_copy<<<blocks2D, threads2D>>>(data + xyz2i(0, by - 1, 0), 1, (bx + 2) * (by + 2),
                                              buf_dev, 1, bx,
                                              bx, bz);
                                             
            cudaMemcpy(buf_back, buf_dev, bx * bz * sizeof(double), cudaMemcpyDeviceToHost);

            MPI_Send(buf_back, bx * bz, MPI_DOUBLE, ijk2i(i2i(id), i2j(id) + 1, i2k(id)), 1, MPI_COMM_WORLD);
        }
        
        if (!is_up)
        {
            /* up -> bufer */
            mat_copy<<<blocks2D, threads2D>>>(data + xyz2i(0, 0, bz - 1), 1, bx + 2,
                                              buf_dev, 1, bx,
                                              bx, by);
                                             
            cudaMemcpy(buf_up, buf_dev, bx * by * sizeof(double), cudaMemcpyDeviceToHost);

            MPI_Send(buf_up, bx * by, MPI_DOUBLE, ijk2i(i2i(id), i2j(id), i2k(id) + 1), 1, MPI_COMM_WORLD);
        }
        
        //==============================================================
        
        if (!is_left)
        {
            MPI_Recv(buf_left, by * bz, MPI_DOUBLE, ijk2i(i2i(id) - 1, i2j(id), i2k(id)), 1, MPI_COMM_WORLD, &status);

            /* left update */
            cudaMemcpy(buf_dev, buf_left, by * bz * sizeof(double), cudaMemcpyHostToDevice);
            
            mat_copy<<<blocks2D, threads2D>>>(buf_dev, 1, by,
                                              data + xyz2i(-1, 0, 0), bx + 2, (bx + 2) * (by + 2),
                                              by, bz);
        }
        
        if (!is_front)
        {
            MPI_Recv(buf_front, bx * bz, MPI_DOUBLE, ijk2i(i2i(id), i2j(id) - 1, i2k(id)), 1, MPI_COMM_WORLD, &status);

            /* front update */
            cudaMemcpy(buf_dev, buf_front, bx * bz * sizeof(double), cudaMemcpyHostToDevice);

            mat_copy<<<blocks2D, threads2D>>>(buf_dev, 1, bx,
                                              data + xyz2i(0, -1, 0), 1, (bx + 2) * (by + 2),
                                              bx, bz);
        }
        
        if (!is_down)
        {
            MPI_Recv(buf_down, bx * by, MPI_DOUBLE, ijk2i(i2i(id), i2j(id), i2k(id) - 1), 1, MPI_COMM_WORLD, &status);

            /* down update */
            cudaMemcpy(buf_dev, buf_down, bx * by * sizeof(double), cudaMemcpyHostToDevice);
            
            mat_copy<<<blocks2D, threads2D>>>(buf_dev, 1, bx,
                                              data + xyz2i(0, 0, -1), 1, bx + 2,
                                              bx, by);
        }

        data_tmp = data;
        data = data_prev;
        data_prev = data_tmp;

        kernel<<<blocks3D, threads3D>>>(data_prev, data,
                                        bx, by, bz,
                                        hx, hy, hz);
        CSC(cudaGetLastError());
                               
        data_diff_and_fabs<<<blocks3D, threads3D>>>(data_prev, data, data_reduce,
                                                    bx, by, bz);
        CSC(cudaGetLastError());
        
        // высчитываем локальный максимум на GPU
        data_ptr = thrust::device_pointer_cast(data_reduce);
        max_u = *thrust::max_element(data_ptr, data_ptr + bx * by * bz);

        // берём глобальный максимум
        MPI_Allreduce(&max_u, &max_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    }
    while(max_global >= eps);

    MPI_Barrier(MPI_COMM_WORLD);
    
    CSC(cudaMemcpy(data_host, data, block_size * sizeof(double), cudaMemcpyDeviceToHost));
    
    /* Подготовка текстового буфера */

    memset(buf_char, ' ', bx * by * bz * N_CHARS_IN_DOUBLE * sizeof(char));
    
    for (z = 0; z < bz; z++)
    {
        for (y = 0; y < by; y++)
        {
            for (x = 0; x < bx; x++)
            {
                sprintf(buf_char + (z * bx * by + y * bx + x) * N_CHARS_IN_DOUBLE,
                        "%7le ", data_host[xyz2i(x, y, z)]);
            }
            if (is_right)
            {
                buf_char[(z * bx * by + y * bx + x) * N_CHARS_IN_DOUBLE - 1] = '\n';
            }
        }
    }

    for (i = 0; i < bx * by * bz * N_CHARS_IN_DOUBLE; i++)
    {
        if (buf_char[i] == '\0')
            buf_char[i] = ' ';
    }

    for(i = 0; i < by * bz; i++)
        blocklens[i] = bx * N_CHARS_IN_DOUBLE;

    for (z = 0; z < bz; z++)
    {
        for (y = 0; y < by; y++)
        {
            indicies[z * by + y] = i2k(id) * px * py * bx * by * bz * N_CHARS_IN_DOUBLE + // global offset
                                   z * px * py * bx * by * N_CHARS_IN_DOUBLE +
                                   i2j(id) * px * bx * by * N_CHARS_IN_DOUBLE +
                                   y * px * bx * N_CHARS_IN_DOUBLE +
                                   i2i(id) * bx * N_CHARS_IN_DOUBLE;
        }
    }
    
    MPI_Type_create_hindexed(by * bz, blocklens, indicies, MPI_CHAR, &filetype);
    MPI_Type_commit(&filetype);

    MPI_File_delete(name_out, MPI_INFO_NULL);

    MPI_File_open(MPI_COMM_WORLD, name_out, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);
    MPI_File_set_view(fp, 0, MPI_CHAR, filetype, "native", MPI_INFO_NULL);
    MPI_File_write_all(fp, buf_char, bx * by * bz * N_CHARS_IN_DOUBLE, MPI_CHAR, &status); // надеюсь, диск выдержит
    MPI_File_close(&fp);
    
    free(MEM_CPU);
    cudaFree(MEM_GPU);
    MPI_Finalize();
    return 0;
}
