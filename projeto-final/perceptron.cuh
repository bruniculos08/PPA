#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <string.h>
#include <iostream>

#define OPTION_OUTPUT true
#define DISCRETE_EVALUATION false
#define EPSILON 0.001
// Para a função de custo: 0 -> mean of abs value differences, 1 -> mean of squared differences 
#define COST_FUNCTION 1
#define TRAINING_TIMES 800000

typedef struct Perceptron perceptron;

struct Perceptron
{
    // Tamanho do vetor weights:
    size_t input_size;
    // O número de pesos é igual ao número de inputs (ou seja, o tamanho da camada anterior):
    double *weights;
    // Constante bias:
    double *b;
};

typedef struct Layer layer;

struct Layer
{
    // Uma camada é um vetor de "perceptrons":
    perceptron *neurons;
    size_t size;
};

typedef struct Network
{
    // Vetor contendo as camadas da rede neural:
    layer *layer_vector;
    // Uma rede tem um número de outputs que diz portanto o número de perceptrons em sua última camada:
    size_t layers_num;
    // Função de ativação:
    double(*activation_function)(double);
} network;

// Função para "imprimir" rede neural densa:
void printDenseNetwork(network &model);
// Função para gerar vetor de dados:
double **readData(char *file_name, int *data_size, int *input_size, int *output_size);
// Função para multiplicar vetores de doubles:
double dotProd(size_t dim, double *v1, double *v2);
// Função para gerar rede com o máximo de perceptrons e conexões entre as camadas:
network genUniformDenseNetwork(size_t layers_num, size_t layer_size, size_t input_size, size_t output_size, double(*activation_function)(double));
// Função para gerar com número personalizado de perceptrons em cada camada 
network genDenseNetwork(size_t layers_num, size_t *layer_size, size_t input_size, size_t output_size, double(*activation_function)(double));
// Função para calcular vetor de output de uma rede neural densa (i.e., todo neurônio da camada i recebe todos os outputs da camada i-1)...
// com base em um vetor de input:
double *evaluateDenseInput(network model, double *input); 
// Função de custo (erro):
double costDenseNetwork(network model, size_t data_size, double **data);
// Função de treino:
void train(network model, size_t data_size, double **data);
// Função para retornar 1 ou -1 de acordo com o sinal de x:
double signal(double x);

// CPU functions to use GPU processing:
double costDenseNetworkUsingGPU(network *model, size_t data_size, double **data);
void trainUsingGPU(network model, size_t data_size, double **data);
double *evaluateDenseInputUsingGPU(network *model, double *input);
void *copyVectorToGPU(void *content, size_t size)
{
    void *address;
    cudaMalloc(&address, size);
    cudaMemcpy(address, content, size, cudaMemcpyHostToDevice);
    return address;
}
void *copyData(void *content, size_t size, cudaMemcpyKind option)
{
    void *address;
    if ((option == cudaMemcpyHostToDevice) || (option == cudaMemcpyDeviceToDevice))
        cudaMalloc(&address, size);
    else if (option != cudaMemcpyDefault)
        address = malloc(size);
    else exit(1);
    cudaMemcpy(address, content, size, option);
    return address;
}

// GPU functions:
__device__ double dotProductDevice(double *A, double *B, size_t lenght);
__global__ void evaluateLayerOutput(network *model, size_t layer_index, double *input, double **output);
