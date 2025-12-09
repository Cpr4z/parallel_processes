#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

double** allocate_matrix(int n) {
    double **matrix = (double**)malloc(n * sizeof(double*));
    if (matrix == NULL) {
        printf("Ошибка выделения памяти для указателей матрицы\n");
        exit(1);
    }
    for (int i = 0; i < n; i++) {
        matrix[i] = (double*)malloc(n * sizeof(double));
        if (matrix[i] == NULL) {
            printf("Ошибка выделения памяти для строки %d матрицы\n", i);
            exit(1);
        }
    }
    return matrix;
}

void free_matrix(double **matrix, int n) {
    for (int i = 0; i < n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void copy_matrix(double **src, double **dst, int n) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dst[i][j] = src[i][j];
        }
    }
}

void multiply_matrices(double **A, double **B, double **result, int n) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i][j] = 0;
            for (int k = 0; k < n; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

bool is_identity_matrix(double **matrix, int n, double tolerance) {
    bool is_identity = true;
    #pragma omp parallel for collapse(2) reduction(&&:is_identity)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double expected = (i == j) ? 1.0 : 0.0;
            if (fabs(matrix[i][j] - expected) > tolerance) {
                is_identity = false;
            }
        }
    }
    return is_identity;
}

void gauss_jordan_omp(double **matrix, double **inverse, int n) {
    double **tmp_matrix = allocate_matrix(n);
    copy_matrix(matrix, tmp_matrix, n);
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            inverse[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }

    for (int pivot = 0; pivot < n; pivot++) {
        int max_row = pivot;
        double max_val = fabs(tmp_matrix[pivot][pivot]);
        
        for (int k = pivot + 1; k < n; k++) {
            double current_val = fabs(tmp_matrix[k][pivot]);
            if (current_val > max_val) {
                max_val = current_val;
                max_row = k;
            }
        }
        
        if (max_row != pivot) {
            #pragma omp parallel for
            for (int j = 0; j < n; j++) {
                double temp = tmp_matrix[pivot][j];
                tmp_matrix[pivot][j] = tmp_matrix[max_row][j];
                tmp_matrix[max_row][j] = temp;
                
                temp = inverse[pivot][j];
                inverse[pivot][j] = inverse[max_row][j];
                inverse[max_row][j] = temp;
            }
        }
        
        if (fabs(tmp_matrix[pivot][pivot]) < 1e-12) {
            free_matrix(tmp_matrix, n);
            return;
        }
        
        double pivot_val = tmp_matrix[pivot][pivot];
        #pragma omp parallel for
        for (int j = 0; j < n; j++) {
            tmp_matrix[pivot][j] /= pivot_val;
            inverse[pivot][j] /= pivot_val;
        }
        
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            if (i != pivot) {
                double factor = tmp_matrix[i][pivot];
                for (int j = 0; j < n; j++) {
                    tmp_matrix[i][j] -= factor * tmp_matrix[pivot][j];
                    inverse[i][j] -= factor * inverse[pivot][j];
                }
            }
        }
    }
    
    free_matrix(tmp_matrix, n);
}

int main(int argc, char *argv[]) {
    int myrank, nprocs;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (argc != 2) {
        if (myrank == 0) {
            printf("Usage: %s <matrix_size>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    int matrix_size = atoi(argv[1]);
    if (matrix_size <= 0) {
        if (myrank == 0) {
            printf("Matrix size must be positive\n");
        }
        MPI_Finalize();
        return 2;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (myrank == 0) {
        FILE *output_file = fopen("results.txt", "w");
        if (output_file == NULL) {
            printf("Error opening output file\n");
            MPI_Abort(MPI_COMM_WORLD, 3);
            return 3;
        }

        double **matrix = allocate_matrix(matrix_size);
        double **inverse = allocate_matrix(matrix_size);
        double **original = allocate_matrix(matrix_size);
        double **product = allocate_matrix(matrix_size);

        srand(time(NULL));
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < matrix_size; i++) {
            for (int j = 0; j < matrix_size; j++) {
                matrix[i][j] = (rand() % 201) - 100;
            }
        }

        #pragma omp parallel for
        for (int i = 0; i < matrix_size; i++) {
            matrix[i][i] += 200;
        }

        copy_matrix(matrix, original, matrix_size);

        double start_time = omp_get_wtime();
        gauss_jordan_omp(matrix, inverse, matrix_size);
        double comp_time = omp_get_wtime() - start_time;

        multiply_matrices(original, inverse, product, matrix_size);
        const double tolerance = 1e-6;

        printf("Matrix size: %d\n", matrix_size);
        printf("MPI Processes count: %d\n", nprocs);
        printf("OpenMP Threads count: %d\n", omp_get_max_threads());
        
        if (is_identity_matrix(product, matrix_size, tolerance)) {
            printf("Result: CORRECT\n");
        } else {
            printf("Result: INCORRECT\n");
        }
        printf("Computation time: %.6f seconds\n", comp_time);
        
        fprintf(output_file, "Matrix size: %d\n", matrix_size);
        fprintf(output_file, "MPI Processes count: %d\n", nprocs);
        fprintf(output_file, "OpenMP Threads count: %d\n", omp_get_max_threads());
        if (is_identity_matrix(product, matrix_size, tolerance)) {
            fprintf(output_file, "Result: CORRECT\n");
        } else {
            fprintf(output_file, "Result: INCORRECT\n");
        }
        fprintf(output_file, "Computation time: %.6f seconds\n", comp_time);

        double max_error = 0;
        int incorrect_count = 0;

        #pragma omp parallel for collapse(2) reduction(max:max_error) reduction(+:incorrect_count)
        for (int i = 0; i < matrix_size; i++) {
            for (int j = 0; j < matrix_size; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                double error = fabs(product[i][j] - expected);
                if (error > max_error) {
                    max_error = error;
                }
                if (error > tolerance) {
                    incorrect_count++;
                }
            }
        }

        printf("Max error: %e\n", max_error);
        printf("Incorrect elements: %d\n", incorrect_count);
        
        fprintf(output_file, "Max error: %e\n", max_error);
        fprintf(output_file, "Incorrect elements: %d\n", incorrect_count);
        fprintf(output_file, "----------------------------------------\n");
        
        fclose(output_file);

        free_matrix(matrix, matrix_size);
        free_matrix(inverse, matrix_size);
        free_matrix(original, matrix_size);
        free_matrix(product, matrix_size);
    } else {
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}