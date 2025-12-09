#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

#define AEL(A, row, col, n) ((A)[(row) * (n) + (col)])

double** allocate_matrix_2d(int n) {
    double **matrix = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        matrix[i] = (double*)calloc(n, sizeof(double));
    }
    return matrix;
}

void free_matrix_2d(double **matrix, int n) {
    for (int i = 0; i < n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void get_block_rows_info(int n, int rank, int size, int *local_rows, int *row_offset) {
    int base = n / size;
    int rem = n % size;
    
    if (rank < rem) {
        *local_rows = base + 1;
        *row_offset = rank * (base + 1);
    } else {
        *local_rows = base;
        *row_offset = rem * (base + 1) + (rank - rem) * base;
    }
}

void gauss_jordan_mpi(int n, double *local_A, double *local_inv,
                             int local_rows, int row_offset,
                             int rank, int size) {
    double *pivot_row_A = (double*)malloc(n * sizeof(double));
    double *pivot_row_inv = (double*)malloc(n * sizeof(double));
    
    int *row_owner = (int*)malloc(n * sizeof(int));
    for (int r = 0; r < size; r++) {
        int lr, ro;
        get_block_rows_info(n, r, size, &lr, &ro);
        for (int i = 0; i < lr; i++) {
            row_owner[ro + i] = r;
        }
    }

    for (int pivot = 0; pivot < n; pivot++) {
        int owner = row_owner[pivot];
        
        if (rank == owner) {
            int local_pivot = pivot - row_offset;
            
            int best_row = local_pivot;
            double max_val = fabs(AEL(local_A, local_pivot, pivot, n));
            
            for (int k = local_pivot + 1; k < local_rows; k++) {
                double val = fabs(AEL(local_A, k, pivot, n));
                if (val > max_val) {
                    max_val = val;
                    best_row = k;
                }
            }
            
            if (best_row != local_pivot) {
                for (int j = 0; j < n; j++) {
                    double temp = AEL(local_A, local_pivot, j, n);
                    AEL(local_A, local_pivot, j, n) = AEL(local_A, best_row, j, n);
                    AEL(local_A, best_row, j, n) = temp;
                    
                    temp = AEL(local_inv, local_pivot, j, n);
                    AEL(local_inv, local_pivot, j, n) = AEL(local_inv, best_row, j, n);
                    AEL(local_inv, best_row, j, n) = temp;
                }
            }
            
            double pivot_val = AEL(local_A, local_pivot, pivot, n);
            
            if (fabs(pivot_val) < 1e-12) {
                if (rank == 0) {
                    printf("Matrix is singular at pivot %d\n", pivot);
                }
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            
            for (int j = 0; j < n; j++) {
                AEL(local_A, local_pivot, j, n) /= pivot_val;
                AEL(local_inv, local_pivot, j, n) /= pivot_val;
            }
            
            for (int j = 0; j < n; j++) {
                pivot_row_A[j] = AEL(local_A, local_pivot, j, n);
                pivot_row_inv[j] = AEL(local_inv, local_pivot, j, n);
            }
        }
        
        MPI_Bcast(pivot_row_A, n, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        MPI_Bcast(pivot_row_inv, n, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        
        for (int local_row = 0; local_row < local_rows; local_row++) {
            int global_row = row_offset + local_row;
            
            if (global_row == pivot) continue;
            
            double factor = AEL(local_A, local_row, pivot, n);
            
            for (int j = 0; j < n; j++) {
                AEL(local_A, local_row, j, n) -= factor * pivot_row_A[j];
                AEL(local_inv, local_row, j, n) -= factor * pivot_row_inv[j];
            }
        }
    }
    
    free(pivot_row_A);
    free(pivot_row_inv);
    free(row_owner);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc != 2) {
        if (rank == 0) {
            printf("Usage: %s <matrix_size>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    int n = atoi(argv[1]);
    if (n <= 0) {
        if (rank == 0) {
            printf("Matrix size must be positive\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    double total_start = MPI_Wtime();
    
    int local_rows, row_offset;
    get_block_rows_info(n, rank, size, &local_rows, &row_offset);
    
    double *local_A = (double*)calloc(local_rows * n, sizeof(double));
    double *local_inv = (double*)calloc(local_rows * n, sizeof(double));
    
    int *sendcounts = NULL;

    int *displs = NULL;

    double *A_flat = NULL;
    
    if (rank == 0) {
        sendcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        A_flat = (double*)malloc(n * n * sizeof(double));
        
        int current_displ = 0;
        for (int r = 0; r < size; r++) {
            int lr, ro;
            get_block_rows_info(n, r, size, &lr, &ro);
            sendcounts[r] = lr * n;
            displs[r] = current_displ;
            current_displ += sendcounts[r];
        }
        
        srand(time(NULL));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A_flat[i * n + j] = (rand() % 201) - 100;
            }
            A_flat[i * n + i] += 200;
        }
    }
    
    MPI_Scatterv(A_flat, sendcounts, displs, MPI_DOUBLE,
                 local_A, local_rows * n, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    
    for (int i = 0; i < local_rows; i++) {
        int global_row = row_offset + i;
        for (int j = 0; j < n; j++) {
            AEL(local_inv, i, j, n) = (global_row == j) ? 1.0 : 0.0;
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double comp_start = MPI_Wtime();
    
    gauss_jordan_mpi(n, local_A, local_inv, local_rows, row_offset, rank, size);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double comp_time = MPI_Wtime() - comp_start;
    
    double *inv_flat = NULL;
    if (rank == 0) {
        inv_flat = (double*)malloc(n * n * sizeof(double));
    }
    
    MPI_Gatherv(local_inv, local_rows * n, MPI_DOUBLE,
                inv_flat, sendcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    double total_time = MPI_Wtime() - total_start;
    
    if (rank == 0) {
        double check_start = MPI_Wtime();
        
        double **A = allocate_matrix_2d(n);
        double **inv = allocate_matrix_2d(n);
        double **prod = allocate_matrix_2d(n);

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] = A_flat[i * n + j];
                inv[i][j] = inv_flat[i * n + j];
            }
        }

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                #pragma omp simd reduction(+:sum)
                for (int k = 0; k < n; k++) {
                    sum += A[i][k] * inv[k][j];
                }
                prod[i][j] = sum;
            }
        }

        double max_error = 0.0;
        int incorrect_count = 0;
        
        #pragma omp parallel for collapse(2) reduction(max:max_error) reduction(+:incorrect_count)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                double error = fabs(prod[i][j] - expected);
                
                if (error > max_error) {
                    max_error = error;
                }
                if (error > 1e-6) {
                    incorrect_count++;
                }
            }
        }

        bool correct = (incorrect_count == 0);
        double check_time = MPI_Wtime() - check_start;
        
        printf("Computation time: %.6f seconds\n", comp_time);
        printf("Max error: %e\n", max_error);
        printf("Result: %s\n", correct ? "CORRECT" : "INCORRECT");

        free_matrix_2d(A, n);
        free_matrix_2d(inv, n);
        free_matrix_2d(prod, n);
        free(A_flat);
        free(inv_flat);
        free(sendcounts);
        free(displs);
    }
    
    MPI_Finalize();
    return 0;
}