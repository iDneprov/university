#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "mpi.h"

#define X 0
#define Y 1
#define Z 2

#define DOWN 0
#define UP 1
#define LEFT 2
#define RIGHT 3
#define FRONT 4
#define BACK 5
#define O 6

// индексация внутри блока
#define BLOCK_3D_TO_1D(x, y, z) (((z) + 1) * (b[X] + 2) * (b[Y] + 2) + ((y) + 1) * (b[X] + 2) + (x) + 1)

// индексация по процессам
#define PROCESS_3D_TO_1D(i, j, k) ((k) * p[X] * p[Y] + (j) * p[X] + (i))
#define GET_K(i) ((i) / (p[X] * p[Y]))
#define GET_I(i) (((i) - GET_K(i) * p[X] * p[Y]) % p[X])
#define GET_J(i) (((i) - GET_K(i) * p[X] * p[Y]) / p[X])

#define MAX(a, b) ((a) < (b) ? (b) : (a))
#define MAX4(a, b, c, d) MAX(MAX(MAX((a), (b)), (c)), (d))

#define BELONGS_TO_DOWN (!GET_K(id))
#define BELONGS_TO_UP (GET_K(id) + 1 == p[Z])
#define BELONGS_TO_LEFT (!GET_I(id))
#define BELONGS_TO_RIGHT (GET_I(id) + 1 == p[X])
#define BELONGS_TO_FRONT (!GET_J(id))
#define BELONGS_TO_BACK (GET_J(id) + 1 == p[Y])

#define vector_print(vec, n)              \
{                                          \
    int i;                                  \
    for (i = 0; i < (n); i++)                \
    {                                         \
        fprintf(output_file, "%7le ", (vec)[i]);      \
    }                                           \
}

int main(int argc, char *argv[])
{
    int id, numproc;
    int i, j, k, x, y, z,
        buffer_size;
    int p[3], b[3]; // размеры сетки процессов и блока для процесса

    void *host_memorry = NULL;
    double *data, *data_prev, *tmp;
    double *buffer, *buffer_left, *buffer_right, *buffer_front, *buffer_back, *buffer_up, *buffer_down;
    double eps, process_max, max_all;
    double d[3], u[7], h[3]; // размер области, ограничения, шаги


    char output_file_name[128];
    FILE *output_file;
    MPI_Status status_mpi;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &numproc); // число процессов в коммуникаторе
    MPI_Comm_rank(MPI_COMM_WORLD, &id); // номер процесса

    if (!id) { // нулевым процессором будем читать и писать
        for (int i = 0; i < 3; i++)
            scanf("%d", &p[i]);
        for (int i = 0; i < 3; i++)
            scanf("%d", &b[i]);
        scanf("%s", output_file_name);
        scanf("%le", &eps);
        for (int i = 0; i < 3; i++)
            scanf("%lf", &d[i]);
        for (int i = 0; i < 7; i++)
            scanf("%lf", &u[i]);

        for (int i = 0; i < 3; i++)
            h[i] = d[i] / (p[i] * b[i]);
    }
    // делимся полученной информацией
    // от одного процесса ко всем
    MPI_Bcast(p, 3, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, 3, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(h, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(u, 7, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(output_file_name, 128, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD); // ждем, пока все процессы дойдут до этой точки

    // выделяем память и бъем ее
    buffer_size = MAX(MAX(b[X] * b[Y], b[Y] * b[Z]), MAX(b[X] * b[Z], p[X] * p[Y] * p[Z]));
    host_memorry = malloc((2 * (b[X] + 2) * (b[Y] + 2) * (b[Z] + 2) + buffer_size + 2 * (b[Y] * b[X] + b[Z] * b[Y] + b[Z] * b[X])) * sizeof(double));
    data = (double *)host_memorry;
    data_prev = data + (b[X] + 2) * (b[Y] + 2) * (b[Z] + 2);
    buffer = data_prev + (b[X] + 2) * (b[Y] + 2) * (b[Z] + 2);
    buffer_left = buffer + buffer_size;
    buffer_right = buffer_left + b[Z] * b[Y];
    buffer_front = buffer_right + b[Z] * b[Y];
    buffer_back = buffer_front + b[Z] * b[X];
    buffer_up = buffer_back + b[Z] * b[X];
    buffer_down = buffer_up + b[Y] * b[X];

    // закидываем в массивы начальные значения
    for(i = 0; i < (b[X] + 2) * (b[Y] + 2) * (b[Z] + 2); i++) {
        data[i] = u[O]; // начальное значение
        data_prev[i] = u[O];
    // прописываем значения на границах
    } if (BELONGS_TO_LEFT) {
        for(z = 0; z < b[Z]; z++)
            for (y = 0; y < b[Y]; y++) {
                data[BLOCK_3D_TO_1D(-1, y, z)] = u[LEFT];
                data_prev[BLOCK_3D_TO_1D(-1, y, z)] = u[LEFT];
            }
    } if (BELONGS_TO_RIGHT) {
        for (z = 0; z < b[Z]; z++)
            for (y = 0; y < b[Y]; y++) {
                data[BLOCK_3D_TO_1D(b[X], y, z)] = u[RIGHT];
                data_prev[BLOCK_3D_TO_1D(b[X], y, z)] = u[RIGHT];
            }
    } if (BELONGS_TO_FRONT) {
        for (z = 0; z < b[Z]; z++)
            for (x = 0; x < b[X]; x++) {
                data[BLOCK_3D_TO_1D(x, -1, z)] = u[FRONT];
                data_prev[BLOCK_3D_TO_1D(x, -1, z)] = u[FRONT];
            }
    } if (BELONGS_TO_BACK) {
        for (z = 0; z < b[Z]; z++)
            for (x = 0; x < b[X]; x++) {
                data[BLOCK_3D_TO_1D(x, b[Y], z)] = u[BACK];
                data_prev[BLOCK_3D_TO_1D(x, b[Y], z)] = u[BACK];
            }
    } if (BELONGS_TO_DOWN) {
        for(y = 0; y < b[Y]; y++)
            for (x = 0; x < b[X]; x++) {
                data[BLOCK_3D_TO_1D(x, y, -1)] = u[DOWN];
                data_prev[BLOCK_3D_TO_1D(x, y, -1)] = u[DOWN];
            }
    } if (BELONGS_TO_UP) {
        for (y = 0; y < b[Y]; y++)
            for (x = 0; x < b[X]; x++) {
                data[BLOCK_3D_TO_1D(x, y, b[Z])] = u[UP];
                data_prev[BLOCK_3D_TO_1D(x, y, b[Z])] = u[UP];
            }
    }

    do {
        // обмениваемся значениями на границах
        if (!BELONGS_TO_LEFT) { // загоняем их в буферы и отправляем
            for (z = 0, k = 0; z < b[Z]; z++)
                for (y = 0; y < b[Y]; y++, k++)
                    buffer_left[k] = data[BLOCK_3D_TO_1D(0, y, z)];
            // сообщение от одного процесса к другому
            MPI_Send(buffer_left, b[Y] * b[Z], MPI_DOUBLE, PROCESS_3D_TO_1D(GET_I(id) - 1, GET_J(id), GET_K(id)), 1, MPI_COMM_WORLD);
        } if (!BELONGS_TO_FRONT) {
            for (z = 0, k = 0; z < b[Z]; z++)
                for (x = 0; x < b[X]; x++, k++)
                    buffer_front[k] = data[BLOCK_3D_TO_1D(x, 0, z)];
            MPI_Send(buffer_front, b[X] * b[Z], MPI_DOUBLE, PROCESS_3D_TO_1D(GET_I(id), GET_J(id) - 1, GET_K(id)), 1, MPI_COMM_WORLD);
        } if (!BELONGS_TO_DOWN) {
            for (y = 0, k = 0; y < b[Y]; y++)
                for (x = 0; x < b[X]; x++, k++)
                    buffer_down[k] = data[BLOCK_3D_TO_1D(x, y, 0)];
            MPI_Send(buffer_down, b[X] * b[Y], MPI_DOUBLE, PROCESS_3D_TO_1D(GET_I(id), GET_J(id), GET_K(id) - 1), 1, MPI_COMM_WORLD);
        }

        if (!BELONGS_TO_RIGHT) { // вынимаем из буферов
            MPI_Recv(buffer_right, b[Y] * b[Z], MPI_DOUBLE, PROCESS_3D_TO_1D(GET_I(id) + 1, GET_J(id), GET_K(id)), 1, MPI_COMM_WORLD, &status_mpi);
            for (z = 0, k = 0; z < b[Z]; z++)
                for (y = 0; y < b[Y]; y++, k++)
                    data[BLOCK_3D_TO_1D(b[X], y, z)] = buffer_right[k];
        } if (!BELONGS_TO_BACK) {
            MPI_Recv(buffer_back, b[X] * b[Z], MPI_DOUBLE, PROCESS_3D_TO_1D(GET_I(id), GET_J(id) + 1, GET_K(id)), 1, MPI_COMM_WORLD, &status_mpi);
            for (z = 0, k = 0; z < b[Z]; z++)
                for (x = 0; x < b[X]; x++, k++)
                    data[BLOCK_3D_TO_1D(x, b[Y], z)] = buffer_back[k];
        } if (!BELONGS_TO_UP) {
            MPI_Recv(buffer_up, b[X] * b[Y], MPI_DOUBLE, PROCESS_3D_TO_1D(GET_I(id), GET_J(id), GET_K(id) + 1), 1, MPI_COMM_WORLD, &status_mpi);
            for (y = 0, k = 0; y < b[Y]; y++)
                for (x = 0; x < b[X]; x++, k++)
                    data[BLOCK_3D_TO_1D(x, y, b[Z])] = buffer_up[k];
        }

        if (!BELONGS_TO_RIGHT) {
            for (z = 0, k = 0; z < b[Z]; z++)
                for (y = 0; y < b[Y]; y++, k++)
                    buffer_right[k] = data[BLOCK_3D_TO_1D(b[X] - 1, y, z)];
            MPI_Send(buffer_right, b[Y] * b[Z], MPI_DOUBLE, PROCESS_3D_TO_1D(GET_I(id) + 1, GET_J(id), GET_K(id)), 1, MPI_COMM_WORLD);
        } if (!BELONGS_TO_BACK) {
            for (z = 0, k = 0; z < b[Z]; z++)
                for (x = 0; x < b[X]; x++, k++)
                    buffer_back[k] = data[BLOCK_3D_TO_1D(x, b[Y] - 1, z)];
            MPI_Send(buffer_back, b[X] * b[Z], MPI_DOUBLE, PROCESS_3D_TO_1D(GET_I(id), GET_J(id) + 1, GET_K(id)), 1, MPI_COMM_WORLD);
        } if (!BELONGS_TO_UP) {
            for (y = 0, k = 0; y < b[Y]; y++)
                for (x = 0; x < b[X]; x++, k++)
                    buffer_back[k] = data[BLOCK_3D_TO_1D(x, y, b[Z] - 1)];
            MPI_Send(buffer_back, b[X] * b[Y], MPI_DOUBLE, PROCESS_3D_TO_1D(GET_I(id), GET_J(id), GET_K(id) + 1), 1, MPI_COMM_WORLD);
        }

        if (!BELONGS_TO_LEFT) {
            MPI_Recv(buffer_left, b[Y] * b[Z], MPI_DOUBLE, PROCESS_3D_TO_1D(GET_I(id) - 1, GET_J(id), GET_K(id)), 1, MPI_COMM_WORLD, &status_mpi);
            for (z = 0, k = 0; z < b[Z]; z++)
                for (y = 0; y < b[Y]; y++, k++)
                    data[BLOCK_3D_TO_1D(-1, y, z)] = buffer_left[k];
        } if (!BELONGS_TO_FRONT) {
            MPI_Recv(buffer_front, b[X] * b[Z], MPI_DOUBLE, PROCESS_3D_TO_1D(GET_I(id), GET_J(id) - 1, GET_K(id)), 1, MPI_COMM_WORLD, &status_mpi);
            for (z = 0, k = 0; z < b[Z]; z++)
                for (x = 0; x < b[X]; x++, k++)
                    data[BLOCK_3D_TO_1D(x, -1, z)] = buffer_front[k];
        } if (!BELONGS_TO_DOWN) {
            MPI_Recv(buffer_down, b[X] * b[Y], MPI_DOUBLE, PROCESS_3D_TO_1D(GET_I(id), GET_J(id), GET_K(id) - 1), 1, MPI_COMM_WORLD, &status_mpi);
            for (y = 0, k = 0; y < b[Y]; y++)
                for (x = 0; x < b[X]; x++, k++)
                    data[BLOCK_3D_TO_1D(x, y, -1)] = buffer_down[k];
        }
        // ждем, пока всё синхронизируется
        MPI_Barrier(MPI_COMM_WORLD);
        // меняем местами предведущие и текущие значения
        tmp = data_prev;
        data_prev = data;
        data = tmp;

        // пересчитываем результат на текущем шаге
        for(z = 0; z < b[Z]; z++)
            for(y = 0; y < b[Y]; y++)
                for(x = 0; x < b[X]; x++)
                    data[BLOCK_3D_TO_1D(x, y, z)] = ((data_prev[BLOCK_3D_TO_1D(x + 1, y, z)] + data_prev[BLOCK_3D_TO_1D(x - 1, y, z)]) / (h[X] * h[X]) + (data_prev[BLOCK_3D_TO_1D(x, y + 1, z)] + data_prev[BLOCK_3D_TO_1D(x, y - 1, z)]) / (h[Y] * h[Y]) + (data_prev[BLOCK_3D_TO_1D(x, y, z + 1)] + data_prev[BLOCK_3D_TO_1D(x, y, z - 1)]) / (h[Z] * h[Z])) / (2 / (h[X] * h[X]) + 2 / (h[Y] * h[Y]) + 2 / (h[Z] * h[Z]));

        // максимум для процесса
        process_max = fabs(data[BLOCK_3D_TO_1D(0, 0, 0)] - data_prev[BLOCK_3D_TO_1D(0, 0, 0)]);
        for (z = 0; z < b[Z]; z++)
            for (y = 0; y < b[Y]; y++)
                for (x = 0; x < b[X]; x++)
                    process_max = MAX(process_max, fabs(data[BLOCK_3D_TO_1D(x, y, z)] - data_prev[BLOCK_3D_TO_1D(x, y, z)]));

        // максимум среди всех процессов
        MPI_Allreduce(&process_max, &max_all, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    } while(max_all >= eps);

    if (id) { // если процесс не нулевой, то он отправлет данные на нулевой процесс
        for (z = 0; z < b[Z]; z++)
            for (y = 0; y < b[Y]; y++) {
                MPI_Send(data + BLOCK_3D_TO_1D(0, y, z), b[X], MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
            }
    } else { // а если нулевой, то пишет в файл
        output_file = fopen(output_file_name, "w");

        for (k = 0; k < p[Z]; k++) // цикл по процессам
            for (z = 0; z < b[Z]; z++) // цикл по слоям
                for (j = 0; j < p[Y]; j++) // цикл по процессам
                    for (y = 0; y < b[Y]; y++) { // цикл по слоям
                        for (i = 0; i < p[X]; i++) { // цикл по процессам
                            if (PROCESS_3D_TO_1D(i, j, k)) {
                                MPI_Recv(buffer, b[X], MPI_DOUBLE, PROCESS_3D_TO_1D(i, j, k), 1, MPI_COMM_WORLD, &status_mpi);
                                vector_print(buffer, b[X]);
                            } else
                                vector_print(data + BLOCK_3D_TO_1D(0, y, z), b[X]);
                        }
                        fprintf(output_file, "\n");
                    }
        fclose(output_file);
    }

    free(host_memorry);
    MPI_Finalize();
    return 0;
}
