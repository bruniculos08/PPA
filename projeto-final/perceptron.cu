#include "perceptron.cuh"

int main()
{
    matrix *a, *b;
    
    a = buildMatrix<double>(1, 3);
    ((double *) a->values)[0] = 1.0;
    ((double *) a->values)[1] = 1.0;
    ((double *) a->values)[2] = 1.0;
    
    b = buildMatrix<double>(3, 1);
    ((double *) b->values)[0] = 1.0;
    ((double *) b->values)[1] = 1.0;
    ((double *) b->values)[2] = 1.0;

    double c = dotProductUsingGPU(a, b);

    std::cout << "dot product = " << c << std::endl;

    return 0;
}

double dotProductUsingGPU(matrix *A, matrix *B)
{
    assert((A->rows == 1 || A->cols == 1) && (B->rows == 1 || B->cols == 1));
    assert((A->rows == B->rows || A->rows == B->cols) || (A->cols == B->rows || A->cols == B->cols));

    size_t lenght = (A->cols > 1) ? A->cols : A->rows;
    size_t bytes_length = sizeof(double) * lenght;

    printMatrix<double>(A);
    printMatrix<double>(B);

    double dot_product = 0.0;

    // Alloc data inside the GPU address space:
    double *A_device;
    double *B_device;
    cudaMalloc(&A_device, bytes_length);
    cudaMemcpy(A_device, ((double *) A->values), bytes_length, cudaMemcpyHostToDevice);
    cudaMalloc(&B_device, bytes_length);
    cudaMemcpy(B_device, ((double *) B->values), bytes_length, cudaMemcpyHostToDevice);
    
    // Call GPU function:
    double *dot_product_device;
    cudaMalloc(&dot_product_device, sizeof(double));
    dim3 block_dim(1, 1);
    dim3 grid_dim(1, 1);
    multiplyMatricesGPU<<<grid_dim, block_dim>>>(A_device, B_device, lenght, dot_product_device);
    cudaMemcpy(&dot_product, dot_product_device, sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    return dot_product;
}

__global__ void multiplyMatricesGPU(double *A, double *B, size_t row_length, double *C)
{
    int row_index = threadIdx.x;
    int column_index = threadIdx.y;

    double calculated_value = 0.0;

    for(int i = 0; i < row_length; i++)
        calculated_value += A[row_index * row_length + i] * B[i * row_index + column_index];

    C[row_index * row_length + column_index] = calculated_value;
}