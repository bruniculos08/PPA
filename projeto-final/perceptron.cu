#include "perceptron-old.cuh"

void printDenseNetwork(network model)
{
    layer *l;
    int layer_index = 0;
    for(l = model.initial_layer; l != NULL; l = l->next)
    {
        printf("layer %i: ", layer_index);
        for(size_t i = 0; i < (size_t) l->height; i++)
        {
            printf("[");
            for (size_t j = 0; j < (size_t) l->neurons[i].input_size; j++)
            {
                printf("w%li = %.4lf", j, l->neurons[i].weights[j]);
                printf(", ");
            }
            printf("w_bias = %.4lf", l->neurons[i].b);
            printf("] ");
            
        }
        layer_index++;
        printf("\n");
    }
}

double **readData(char *file_name, int *data_size, int *input_size, int *output_size)
{
    FILE *fptr;
    double **data;
    fptr = fopen(file_name, "r");
    char buffer[1024];
    if(fgets(buffer, 1024, fptr))
    {
        while(buffer[0] == '#') fgets(buffer, 1024, fptr);
        sscanf(buffer, "%i %i %i", data_size, input_size, output_size);
        data = (double **) malloc(sizeof(double *) * (*data_size));
        for(size_t i = 0; i < (*data_size); i++)
        {
            data[i] = (double *) malloc(sizeof(double) * ((*input_size) + (*output_size)));
        }
        printf("data stuff: %i %i %i\n", (*data_size), (*input_size), (*output_size));

        int line = 0, column = 0, buffer_index = 0;
        fgets(buffer, 1024, fptr);
        while(line < (*data_size))
        {
            if(buffer[buffer_index] == '#') fgets(buffer, 1024, fptr);
            else if((buffer[buffer_index] == '\n') || (buffer[buffer_index] == '\0'))
            {
                fgets(buffer, 1024, fptr);
                line++;
                column = 0;
                buffer_index = 0;
            }
            else if(buffer[buffer_index] == ' ') buffer_index++;
            else
            {
                sscanf(buffer + buffer_index, "%lf %*s", &data[line][column]);
                column++;
                while((buffer[buffer_index] != ' ') && (buffer[buffer_index] != '\n') && (buffer[buffer_index] != '\0')) 
                {
                    buffer_index++;
                }
            }
        }
    }
    fclose(fptr);
    return data;
}

void trainDenseNetwork(network model, int data_size, double **data)
{
    layer *actual_layer;
    for(actual_layer = model.initial_layer; actual_layer != NULL; actual_layer = actual_layer->next)
    {
        for(size_t i = 0; i < actual_layer->height; i++)
        {
            for(size_t j = 0; j < actual_layer->neurons[i].input_size; j++)
            {   
                // printf("Training w_%li,%li...\n", i, k + 1);
                double cost_w = costDenseNetwork(model, data_size, data); 
                // printf("cost_w = %lf\n", cost_w);
                
                ((actual_layer->neurons[i]).weights[j]) += EPSILON;
                double cost_w_epsilon = costDenseNetwork(model, data_size, data);
                ((actual_layer->neurons[i]).weights[j]) -= EPSILON;

                ((actual_layer->neurons[i]).weights[j]) -= (cost_w_epsilon - cost_w) * EPSILON;
            }
            // To change the bias value:
            double cost_w = costDenseNetwork(model, data_size, data); 
            
            ((actual_layer->neurons[i]).b) += EPSILON;
            double cost_w_epsilon = costDenseNetwork(model, data_size, data);
            ((actual_layer->neurons[i]).b) -= EPSILON;

            // ((actual_layer->neurons[i]).b) -= signal(cost_w_epsilon - cost_w) * EPSILON;
            ((actual_layer->neurons[i]).b) -= (cost_w_epsilon - cost_w) * EPSILON;
        }
    }
}

double signal(double x)
{
        if(x >= 0) return 1.0;
        return -1.0;
}

double costDenseNetwork(network model, size_t data_size, double **data)
{
    size_t input_size = model.input_size;
    size_t output_size = model.output_size;
    double carry = 0.0;
    // data_size is the number of samples and then the cost function is the average of error related to each sample:
    for(size_t i = 0; i < data_size; i++)
    {
        // data[i] = {x1, x2, ..., xn, y1, y2, ..., yk}
        for(size_t j = 0; j < output_size; j++)
        {
            if(COST_FUNCTION == 0)
            {
                carry += fabs((data[i][input_size + j] - (evaluateDenseInput(model, data[i]))[j])) / ((double) data_size);
            }
            // else if(COST_FUNCTION == 1)
            else
            {
                carry += pow((data[i][input_size + j] - (evaluateDenseInput(model, data[i]))[j]), 2.0) / ((double) data_size);
            }
        }
    }
    return carry;
}

double dotProd(size_t dim, double *v1, double *v2)
{
    double carry = 0.0;
    for(size_t i = 0; i < dim; i++)
        carry += (v1[i] * v2[i]);
    return carry;
}

__global__ void dotProdGPU(size_t dim, double *v1, double *v2, double *carry) {
    
}

double *evaluateDenseInput(network model, double *input)
{   
    layer *actual_layer;
    actual_layer = model.initial_layer;
    double *actual_layer_output;
    double *last_layer_output;
    actual_layer_output = NULL;
    // last_layer_output = NULL;
    last_layer_output = input;
    int last_layer_size = model.input_size;
    for(size_t i = 0; i < (size_t) model.layers_num; i++)
    {
        // The problem is that we need to fix it for the initial case when...
        actual_layer_output = (double *) malloc(actual_layer->height * sizeof(double));
        for(size_t j = 0; j < (size_t) actual_layer->height; j++)
        {
            if(model.activation_function != NULL)
            {
                actual_layer_output[j] = (*model.activation_function)(dotProd(last_layer_size, actual_layer->neurons[j].weights, last_layer_output)
                                        + actual_layer->neurons[j].b);
            }
            else 
            actual_layer_output[j] = dotProd(last_layer_size, actual_layer->neurons[j].weights, last_layer_output) + actual_layer->neurons[j].b;
        }
        if(last_layer_output != input) 
            free(last_layer_output);
        last_layer_output = actual_layer_output;
        last_layer_size = actual_layer->height;
    }
    if(DISCRETE_EVALUATION)
    {
        for(size_t i = 0; i < model.output_size; i++)
            last_layer_output[i] = round(last_layer_output[i]);
    }
    return last_layer_output;
}

network genDenseNetwork(size_t layers_num, size_t *layers_size, size_t input_size, size_t output_size, double(*activation_function)(double))
{
    assert(layers_size != NULL);
    assert(layers_num != 0);
    network model;
    model.initial_layer = NULL;
    model.final_layer = NULL;
    model.input_size = input_size;
    model.output_size = output_size;
    model.layers_num = layers_num;
    model.activation_function = activation_function;
    // it should always be the case that layer_size[layers_num - 1] == output_size,...
    // ... otherwise we have the 2 following options:
    //  (1) change the value of layer_size[layers_num - 1] to outputsize;
    //  (2) add one layer, changing layers_num to layers_num + 1, and storing at &layers_size[layers_num - 1]...
    //  ... the value of output_size.
    //  Implementing the last option first:
    if(OPTION_OUTPUT && (layers_size[layers_num - 1] != output_size))
    layers_num++;
    model.layers_size = (size_t *) malloc(sizeof(size_t) * layers_num);
    memcpy((void *) model.layers_num, (void *) layers_size, (size_t) layers_num - 1);
    model.layers_size[layers_num - 1] = output_size;
    
    layer *l = (layer *) malloc(sizeof(layer));
    model.initial_layer = l;
    l->previous = NULL;
    for(size_t i = 0; i < (size_t) layers_num; i++)
    {
        l->height = model.layers_size[i];
        l->neurons = (perceptron *) malloc(sizeof(perceptron) * l->height);
        for(size_t j = 0; j < l->height; j++)
        {
            l->neurons[j].b = 1.0;
            l->neurons[j].input = NULL;
            if(l->previous)
                l->neurons[j].input_size = l->previous->height;
            else
                l->neurons[j].input_size = input_size;
            l->neurons[j].weights = (double *) malloc(l->neurons[j].input_size * sizeof(double));
            for(size_t k = 0; k < l->neurons[j].input_size; k++)
                l->neurons[j].weights[k] = 1.0; 
        }
        if(i < layers_num - 1)
        {
            l->next = (layer *) malloc(sizeof(layer));
            l = l->next;
        }
    }
    model.final_layer = l;
    return model;
}

network genUniformDenseNetwork(size_t layers_num, size_t layer_size, size_t input_size, size_t output_size, double(*activation_function)(double))
{
    network model;
    model.initial_layer = NULL;
    model.final_layer = NULL;
    model.input_size = input_size;
    model.output_size = output_size;
    model.layers_num = layers_num;
    model.activation_function = activation_function;
    layer *l;
    for(size_t i = 0; i < (size_t) layers_num; i++)
    {
        // Caso da primeira camada:
        if(i == 0)
        {
            l = (layer *) malloc(sizeof(layer));
            model.initial_layer = l;
            // Se há mais de uma camada: 
            if(layers_num > 1)
                l->height = layer_size;
            else
                l->height = output_size;
            l->neurons = (perceptron *) malloc((l->height) * sizeof(perceptron));
            l->previous = NULL;
            l->next = NULL;
            for(size_t j = 0; j < (size_t) l->height; j++)
            {
                l->neurons[j].input_size = input_size;
                l->neurons[j].b = 1.0;
                l->neurons[j].weights = (double *) malloc((l->neurons[j].input_size) * sizeof(double));
                for(size_t k = 0; k < (size_t) l->neurons[j].input_size; k++)
                    l->neurons[j].weights[k] = 1.0;
                l->neurons[j].input = NULL;
            }
            continue;
        }
        // Se a atual camada a ser criada não é a última:
        else if(i != (layers_num - 1))
        {
            l->next = (layer *) malloc(sizeof(layer));
            l->next->next = NULL;
            l->next->previous = l;
            l = l->next; 
            l->height = layer_size;
            l->neurons = (perceptron *) malloc(layer_size * sizeof(perceptron));
            for(size_t j = 0; j < (size_t) l->height; j++)
            {
                l->neurons[j].input_size = (l->previous)->height;
                l->neurons[j].b = 0.0;
                l->neurons[j].weights = (double *) malloc(l->neurons[j].input_size * sizeof(double));
                l->neurons[j].input = (perceptron **) malloc(l->neurons[j].input_size * sizeof(perceptron *));
                for(size_t k = 0; k < (size_t) l->neurons[j].input_size; k++)
                {
                    l->neurons[j].weights[k] = 1.0;
                    (l->neurons[j]).input[k] = &(l->previous->neurons[k]);
                }
            }
            continue;
        }
        else
        {
            l->next = (layer *) malloc(sizeof(layer));
            l->next->next = NULL;
            l->next->previous = l;
            l = l->next; 
            model.final_layer = l;
            // A última camada deve ter o mesmo número de neurônios que o output:
            l->height = output_size;
            l->neurons = (perceptron *) malloc(output_size * sizeof(perceptron));
            for(size_t j = 0; j < (size_t) l->height; j++)
            {
                l->neurons[j].input_size = l->previous[0].height;
                l->neurons[j].b = 0.0;
                l->neurons[j].weights = (double *) malloc(l->neurons[j].input_size * sizeof(double));
                l->neurons[j].input = (perceptron **) malloc(l->neurons[j].input_size * sizeof(perceptron *));
                for(size_t k = 0; k < (size_t) l->neurons[j].input_size; k++)
                {
                    l->neurons[j].weights[k] = 1.0;
                    l->neurons[j].input[k] = &l->previous->neurons[k];
                }
            }
            continue;
        }
    }
    if(model.final_layer == NULL) model.final_layer = model.initial_layer;
    return model;
}

double fabsCos(double x)
{
    return fabs(cos(x));
}

double logistic(double x)
{
    return (1 / (1 + exp(-x)));
}

void validateDenseNeuralNetwork(network model, double **data, size_t data_size, size_t input_size, size_t output_size)
{
    printf("evaluating validation dataset:\n");
    for(size_t i = 0; i < data_size; i++)
    {
        double *result = evaluateDenseInput(model, data[i]);
        for(size_t j = 0; j < output_size; j++)
        {
            printf("y%li = %lf", j, result[j]);
            if(j < output_size - 1)
                printf(", ");
        }
        printf("\n");
        free(result);
    }
}

int main(void)
{
    double **data;
    int data_size, input_size, output_size;
    // data = readData("datasets/logistic_regression.txt", &data_size, &input_size, &output_size);
    // data = readData("datasets/data.txt", &data_size, &input_size, &output_size);
    // data = readData("datasets/sub.txt", &data_size, &input_size, &output_size);
    data = readData("datasets/sub_plus_one.txt", &data_size, &input_size, &output_size);
    // data = readData("datasets/xor.txt", &data_size, &input_size, &output_size);
    // for(size_t i = 0; i < (size_t) data_size; i++)
    // {
    //     for(size_t j = 0; j < (size_t) (input_size + output_size); j++)
    //         printf("%lf ", data[i][j]);
    //     printf("\n");
    // }

    // double input_test[] = {1, 1};
    double input_test[] = {2, 1};
    size_t layers_num = 1;
    size_t uniform_layers_size = 1;
    size_t layers_size[] = {output_size}; 
    double **validation = (double **) malloc(sizeof(double *));
    validation[0] = (double *) malloc(2 * sizeof(double));
    validation[0][0] = 2;
    validation[0][1] = 1;

    // network model = genUniformDenseNetwork(layers_num, (size_t) layers_size, (size_t) input_size, (size_t) output_size, NULL);
    // network model = genUniformDenseNetwork(layers_num, uniform_layers_size, (size_t) input_size, (size_t) output_size, logistic);
    // network model = genUniformDenseNetwork(layers_num, (size_t) layers_size, (size_t) input_size, (size_t) output_size, fabsCos);
    
    // network model = genDenseNetwork(layers_num, layers_size, input_size, output_size, logistic);
    network model = genDenseNetwork(layers_num, layers_size, input_size, output_size, NULL);
    printDenseNetwork(model);
    printf("cost function value before %i trains: %lf\n", TRAINING_TIMES, costDenseNetwork(model, (size_t) data_size, data));
    validateDenseNeuralNetwork(model, validation, 1, input_size, output_size);
    // printf("evaluate before %i trainings: %lf %lf\n", TRAINING_TIMES, evaluateDenseInput(model, input_test)[0], evaluateDenseInput(model, input_test)[1]);
    for(size_t i = 0; i < (size_t) TRAINING_TIMES; i++) trainDenseNetwork(model, (size_t) data_size, data);
    printf("cost function value after %i trains: %lf\n", TRAINING_TIMES, costDenseNetwork(model, (size_t) data_size, data));
    validateDenseNeuralNetwork(model, validation, 1, input_size, output_size);
    // printf("evaluate after %i trainings: %lf %lf\n", TRAINING_TIMES, evaluateDenseInput(model, input_test)[0], evaluateDenseInput(model, input_test)[1]);
    printDenseNetwork(model);
}