#include <stdio.h>
#include <cassert>
#include <iostream>

// Struct for linear stored matrices:
typedef struct Matrix
{
    size_t rows, cols, data_size;
    void *values;
} matrix;

// GPU functions:
//  - Multiplying linear stored matrices using GPU:
__global__ void multiplyMatricesGPU(double *A, double *B, size_t row_length, double *C);

// CPU functions:
//  - Print matrix:
template <typename T>
void printMatrix(matrix *A)
{
    for(size_t row = 0; row < A->rows; row++)
    {
        for(size_t col = 0; col < A->cols; col++)
            std::cout << ((T *) A->values)[row * A->cols + col];
        std::cout << std::endl;
    }
}
//  - Build matrix inside CPU address space:
template <typename T>
matrix *buildMatrix(size_t rows, size_t cols)
{
    matrix *a = (matrix *) malloc(sizeof(matrix));
    a->rows = rows;
    a->cols = cols;
    a->data_size = sizeof(T);
    a->values = new T[rows * cols];
    return a;
}
//  - Multiply matrices using GPU:
double dotProductUsingGPU(matrix *A, matrix *B);

typedef struct Perceptron
{
    // Vector for weights:
    matrix weights;
    // Obs.: note that the value weight.rows is the perceptron input size. 
    // Perceptron bias:
    double b;
} perceptron;

typedef struct Layer layer;

struct Layer
{
    // A layer contains a vector of perceptrons:
    perceptron *neurons;
    int size;
    // The layers are a chained list:
    layer *previous;
    layer *next;
};

typedef struct Network
{
    size_t input_size;
    size_t output_size;
    size_t layers_num;
    size_t *layers_size;
    layer *initial_layer;
    layer *final_layer;
    double(*activation_function)(double);
} network;

