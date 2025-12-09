#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>

double** allocate_matrix(int n) {
    double **matrix = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        matrix[i] = (double*)malloc(n * sizeof(double));
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
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dst[i][j] = src[i][j];
        }
    }
}

void multiply_matrices(double **A, double **B, double **result, int n) {
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
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double expected = (i == j) ? 1.0 : 0.0;
            if (fabs(matrix[i][j] - expected) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

void print_matrix(double **matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%10.6f ", matrix[i][j]);
        }
        printf("\n");
    }
}

void gauss_jordan(double **matrix, double **inverse, int n) {
    double **tmp_matrix = allocate_matrix(n);
    copy_matrix(matrix, tmp_matrix, n);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            inverse[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }

    for (int i = 0; i < n; i++) {
        int max_row = i;
        for (int k = i + 1; k < n; k++) {
            if (fabs(tmp_matrix[k][i]) > fabs(tmp_matrix[max_row][i])) {
                max_row = k;
            }
        }
        
        if (max_row != i) {
            for (int j = 0; j < n; j++) {
                double temp = tmp_matrix[i][j];
                tmp_matrix[i][j] = tmp_matrix[max_row][j];
                tmp_matrix[max_row][j] = temp;
                
                temp = inverse[i][j];
                inverse[i][j] = inverse[max_row][j];
                inverse[max_row][j] = temp;
            }
        }
        
        if (fabs(tmp_matrix[i][i]) < 1e-10) {
            printf("Матрица вырожденная или почти вырожденная\n");
            free_matrix(tmp_matrix, n);
            return;
        }
        
        double pivot = tmp_matrix[i][i];
        for (int j = 0; j < n; j++) {
            tmp_matrix[i][j] /= pivot;
            inverse[i][j] /= pivot;
        }
        
        for (int k = 0; k < n; k++) {
            if (k != i) {
                double factor = tmp_matrix[k][i];
                for (int j = 0; j < n; j++) {
                    tmp_matrix[k][j] -= factor * tmp_matrix[i][j];
                    inverse[k][j] -= factor * inverse[i][j];
                }
            }
        }
    }
    free_matrix(tmp_matrix, n);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("В программу необходимо передать размер матрицы\n");
        return 1;
    }

    int matrix_size = atoi(argv[1]);
    if (matrix_size <= 0) {
        printf("Размер матрицы должен быть > 0\n");
        return 2;
    }

    double **matrix = allocate_matrix(matrix_size);
    double **inverse = allocate_matrix(matrix_size);
    double **original = allocate_matrix(matrix_size);
    double **product = allocate_matrix(matrix_size);

    srand(time(NULL));
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            matrix[i][j] = (rand() % 201) - 100;
        }
        matrix[i][i] += 200;
    }

    copy_matrix(matrix, original, matrix_size);

    clock_t start = clock();
    gauss_jordan(matrix, inverse, matrix_size);
    clock_t end = clock(); 
    double comp_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Matrix size: %i", matrix_size);
    printf("Computation time: %.6f seconds\n", comp_time);

    multiply_matrices(original, inverse, product, matrix_size);
    
    const double tolerance = 1e-6;
    if (is_identity_matrix(product, matrix_size, tolerance)) {
        printf("Result: CORRECT\n");
    } else {
        printf("Result: INCORRECT\n");
        double max_error = 0;
        for (int i = 0; i < matrix_size; i++) {
            for (int j = 0; j < matrix_size; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                double error = fabs(product[i][j] - expected);
                if (error > max_error) {
                    max_error = error;
                }
            }
        }
        printf("Max error: %e\n", max_error);
    }

    free_matrix(matrix, matrix_size);
    free_matrix(inverse, matrix_size);
    free_matrix(original, matrix_size);
    free_matrix(product, matrix_size);

    return 0;
}