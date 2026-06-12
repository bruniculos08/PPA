#include <stdio.h>
#include <cassert>

typedef struct Matrix
{
    size_t rows, cols;
    double *values;
} matrix;

__global__ void simpleFunction(void) {
    printf("Testing\n");
}

__global__ void multiplyMatricesGPU(double *a, double *b, size_t row_length, double *c)
{
    int row_index = threadIdx.x;
    int column_index = threadIdx.y;
    
    double calculated_value = 0;

    for(int i = 0; i < row_length; i++)
        calculated_value += a[row_index * row_length + i] * b[i * row_index + column_index];

    c[row_index * row_length + column_index] = calculated_value;
}

void printMatrix(matrix *a)
{
    for(size_t row = 0; row < a->rows; row++)
    {
        for(size_t col = 0; col < a->cols; col++)
            printf("%lf ", a->values[row * a->cols + col]);
        printf("\n");
    }
}

matrix *buildMatrix(size_t rows, size_t cols)
{
    matrix *a = (matrix *) malloc(sizeof(matrix));
    a->rows = rows;
    a->cols = cols;
    a->values = new double[rows * cols];
    return a;
}

matrix *multiplyMatrices(matrix *a, matrix *b)
{
    assert(a->cols == b->rows);
        
    double *a_device_values, *b_device_values, *c_device_values;
    
    cudaMalloc((void **) &a_device_values, sizeof(double) * a->rows * a->cols);
    cudaMemcpy((void *) a_device_values, a->values, sizeof(double) * a->rows * a->cols, cudaMemcpyHostToDevice);
    
    cudaMalloc((void **) &b_device_values, sizeof(double) * b->rows * b->cols);
    cudaMemcpy((void *) b_device_values, b->values, sizeof(double) * b->rows * b->cols, cudaMemcpyHostToDevice);
    
    matrix *c = buildMatrix(a->rows, b->cols);
    cudaMalloc((void **) &c_device_values, sizeof(double) * c->rows * c->cols);
    
    dim3 block_dim(a->rows, b->cols);
    dim3 grid_dim(1, 1);
    
    multiplyMatricesGPU<<<grid_dim, block_dim>>>(a_device_values, b_device_values, a->cols, c_device_values);
    
    cudaMemcpy((void *) c->values, c_device_values, sizeof(double) * c->rows * c->cols, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    return c;
}

int main() {

    matrix *a, *b;
    
    a = buildMatrix(1, 3);
    a->values[0] = 1.0;
    a->values[1] = 1.0;
    a->values[2] = 1.0;
    
    b = buildMatrix(3, 1);
    b->values[0] = 1.0;
    b->values[1] = 1.0;
    b->values[2] = 1.0;

    matrix *c = multiplyMatrices(a, b);

    printMatrix(c);

    return 0;
}
