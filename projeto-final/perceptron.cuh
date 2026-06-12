#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <string.h>

#define OPTION_OUTPUT true
#define DISCRETE_EVALUATION false
#define EPSILON 0.001
// Para a função de custo: 0 -> mean of abs value differences, 1 -> mean of squared differences 
#define COST_FUNCTION 1
#define TRAINING_TIMES 800000

typedef struct Perceptron perceptron;

struct Perceptron
{
    // Os inputs de um "perceptron" são os outputs dos "perceptrons" da camanda...
    // ... anterior ligados e ele:
    perceptron **input;
    int input_size;
    // O número de pesos é igual ao número de inputs:
    double *weights;
    // Constante bias:
    double b;
};

typedef struct Layer layer;

struct Layer
{
    // Uma camada é um vetor de "perceptrons":
    perceptron *neurons;
    int height;
    // As camadas serão colocadas em uma lista encadeada e portanto são necessárias...
    // ... as seguintes informações:
    layer *previous;
    layer *next;
};

typedef struct Network
{
    size_t input_size;
    // Uma rede tem um número de outputs que diz portanto o número de perceptrons em sua última camada:
    size_t output_size;
    size_t layers_num;
    size_t *layers_size;
    // Camada inicial:
    layer *initial_layer;
    // Camada final:
    layer *final_layer;
    // Função de ativação:
    double(*activation_function)(double);
} network;

// Função para "imprimir" rede neural densa:
void printDenseNetwork(network model);
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
// Função para colocar k no intervalo [low,high]:
int clampi(int k, int low, int high);
// Função para retornar 1 ou -1 de acordo com o sinal de x:
double signal(double x);
// Função para calcular área de um triângulo de vértices (xa,ya), (xb,yb) e (xc,yc):
double triangleArea(double xa, double ya, double xb, double yb, double xc, double yc);
// Função para verificar se um ponto (xp,yp) está contido no triângulo de vértices (xa,ya), (xb,yb) e (xc,yc):
bool pointInsideTriangle(double xp, double yp, double xa, double ya, double xb, double yb, double xc, double yc);
// Desenha em uma matriz "input" um retângulo de dimensões width e height:
void generateRect(double **input, size_t input_width, size_t input_height, double value, size_t x0, size_t y0, size_t width, size_t height);