#include "perceptron.cuh"

void printDenseNetwork(network &model)
{
    for (size_t i = 0; i < model.layers_num; i++)
    {
        layer l = model.layer_vector[i];
        std::cout << "layer " << i << ":" << std::endl;
        for (size_t j = 0; j < (size_t) l.size; j++)
        {
            printf("[");
            for (size_t j = 0; j < (size_t) l.neurons[i].input_size; j++)
            {
                printf("w%li = %.4lf", j, l.neurons[i].weights[j]);
                printf(", ");
            }
            printf("w_bias = %.4lf", l.neurons[i].b);
            printf("] ");
            
        }
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
    for(size_t layer_index = 0; layer_index < model.layers_num; layer_index++)
    {
        layer actual_layer = model.layer_vector[layer_index];
        for(size_t i = 0; i < actual_layer.size; i++)
        {
            for(size_t j = 0; j < actual_layer.neurons[i].input_size; j++)
            {   
                // printf("Training w_%li,%li...\n", i, k + 1);
                double cost_w = costDenseNetwork(model, data_size, data); 
                // printf("cost_w = %lf\n", cost_w);
                
                ((actual_layer.neurons[i]).weights[j]) += EPSILON;
                double cost_w_epsilon = costDenseNetwork(model, data_size, data);
                ((actual_layer.neurons[i]).weights[j]) -= EPSILON;

                ((actual_layer.neurons[i]).weights[j]) -= (cost_w_epsilon - cost_w) * EPSILON;
            }
            // To change the bias value:
            double cost_w = costDenseNetwork(model, data_size, data); 
            
            ((actual_layer.neurons[i]).b) += EPSILON;
            double cost_w_epsilon = costDenseNetwork(model, data_size, data);
            ((actual_layer.neurons[i]).b) -= EPSILON;

            // ((actual_layer->neurons[i]).b) -= signal(cost_w_epsilon - cost_w) * EPSILON;
            ((actual_layer.neurons[i]).b) -= (cost_w_epsilon - cost_w) * EPSILON;
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
    size_t input_size = model.layer_vector[0].size;
    size_t output_size = model.layer_vector[model.layers_num - 1].size;
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

__device__ double dotProdGPU(size_t dim, double *v1, double *v2)
{
    double carry = 0.0;
    for(size_t i = 0; i < dim; i++)
        carry += (v1[i] * v2[i]);
    return carry;
}

double *evaluateDenseInput(network model, double *input)
{   
    layer *actual_layer;
    actual_layer = &model.layer_vector[0];
    double *actual_layer_output;
    double *last_layer_output;
    actual_layer_output = NULL;
    // last_layer_output = NULL;
    last_layer_output = input;
    int last_layer_size = model.layer_vector[0].size;
    for(size_t i = 0; i < (size_t) model.layers_num; i++)
    {
        // The problem is that we need to fix it for the initial case when...
        actual_layer_output = (double *) malloc(actual_layer->size * sizeof(double));
        for(size_t j = 0; j < (size_t) actual_layer->size; j++)
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
        last_layer_size = actual_layer->size;
    }
    if(DISCRETE_EVALUATION)
    {
        for(size_t i = 0; i < model.layer_vector[model.layers_num - 1].size; i++)
            last_layer_output[i] = round(last_layer_output[i]);
    }
    return last_layer_output;
}

__global__ void evaluateLayerOutput(network *model, size_t layer_index, double *input, double **output)
{
    size_t index = threadIdx.y;
    layer *l = &model->layer_vector[0];
    printf("Computing value for index = %i and layer index = %i\n", index, layer_index);
    size_t output_size = (layer_index >= 1) ? model->layer_vector[layer_index - 1].size : model->layer_vector[0].neurons[0].input_size; 
    cudaMalloc(output, output_size * sizeof(double));
    perceptron *neuron = &l->neurons[index];
    if(model->activation_function != NULL)
        (*output)[index] = (*model->activation_function)(dotProdGPU(l->size, neuron->weights, input) + neuron->b);
    else
        (*output)[index] = (dotProdGPU(l->size, neuron->weights, input) + neuron->b);
    printf("Computed value for index = %i and layer index = %i: %lf\n", index, layer_index, (*output)[index]);
}

double *evaluateDenseInputInsideGPU(network *model, double *input)
{
    double *layer_input = input;
    double *layer_output;
    network *model_info = (network *) malloc(sizeof(network));
    cudaMemcpy(model_info, model, sizeof(network), cudaMemcpyDeviceToHost);
    layer *layer_vector;
    cudaMemcpy(layer_vector, model_info->layer_vector, model_info->layers_num * sizeof(layer), cudaMemcpyDeviceToHost);
    printf("model_info->layers_num = %i\n", model_info->layers_num);
    for (size_t i = 0; i < model_info->layers_num; i++)
    {
        dim3 grid_dim(1);
        dim3 block_dim(1, layer_vector[i].size);
        evaluateLayerOutput<<<grid_dim,block_dim>>>(model, i, layer_input, &layer_output);
        // cudaFree(layer_input);
        layer_input = layer_output;
    }
    cudaDeviceSynchronize();
    size_t output_size = layer_vector[model_info->layers_num - 1].size;
    double *result = (double *) malloc(output_size * sizeof(double));
    cudaMemcpy(result, layer_output, output_size * sizeof(double), cudaMemcpyDeviceToHost);
    return result;
}

network *copyNetworkToGPU(network model)
{
    for (size_t i = 0; i < model.layers_num; i++)
    {
        layer *l = &model.layer_vector[i];
        for (size_t j = 0; j < l->size; j++)
        {
            perceptron *p = &l->neurons[i];
            p->weights = (double *) copyVectorToGPU(p->weights, p->input_size * sizeof(double));
        }
        l->neurons = (perceptron *) copyVectorToGPU(l->neurons, l->size * sizeof(perceptron));
    }
    model.layer_vector = (layer *) copyVectorToGPU(model.layer_vector, model.layers_num * sizeof(layer));
    network *model_copy = (network *) copyVectorToGPU(&model, sizeof(network));
    return model_copy;
}

network genUniformDenseNetwork(size_t layers_num, size_t layer_size, size_t input_size, size_t output_size, double(*activation_function)(double))
{
    network model;
    model.layers_num = layers_num;
    model.layer_vector = (layer *) malloc(layers_num * sizeof(layer));
    for (size_t i = 0; i < layers_num; i++)
    {
        layer *l = &model.layer_vector[i];
        l->size = layer_size;
        l->neurons = (perceptron *) malloc(layer_size * sizeof(perceptron));
        for (size_t j = 0; j < layer_size; j++)
        {
            l->neurons[i].b = 0.0;
            l->neurons[i].input_size = (i >= 1) ? model.layer_vector[i - 1].size : input_size;
            l->neurons[i].weights = (double *) malloc(l->neurons[i].input_size * sizeof(double));
            memset(l->neurons[i].weights, 0, l->neurons[i].input_size * sizeof(double));
        }
    }
    model.activation_function = activation_function;
    return model;
}

network genDenseNetwork(size_t layers_num, size_t *layer_size, size_t input_size, size_t output_size, double(*activation_function)(double))
{
    network model;
    model.layers_num = layers_num;
    model.layer_vector = (layer *) malloc(layers_num * sizeof(layer));
    for (size_t i = 0; i < layers_num; i++)
    {
        layer *l = &model.layer_vector[i];
        l->size = layer_size[i];
        l->neurons = (perceptron *) malloc(layer_size[i] * sizeof(perceptron));
        for (size_t j = 0; j < layer_size[i]; j++)
        {
            l->neurons[j].b = 1.0;
            l->neurons[j].input_size = (i >= 1) ? model.layer_vector[i - 1].size : input_size;
            l->neurons[j].weights = (double *) malloc(l->neurons[j].input_size * sizeof(double));
            for (size_t weight_index = 0; weight_index < l->neurons[i].input_size; weight_index++)
                l->neurons[i].weights[weight_index] = 1.0;
        }
    }
    model.activation_function = activation_function;
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

    double *output_evaluated_by_CPU = evaluateDenseInput(model, input_test);

    std::cout << "GPU tests:" << std::endl;
    network *model_GPU_address = copyNetworkToGPU(model);
    double *output;
    cudaMalloc(&output, output_size * sizeof(double));
    evaluateDenseInputInsideGPU(model_GPU_address, input_test);
    double *output_CPU = (double *) malloc(output_size * sizeof(double));
    cudaMemcpy(output_CPU, output, output_size * sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "evaluation result:" << std::endl;
    for (size_t i = 0; i < output_size; i++)
    {
        std::cout << output_CPU[i] << std::endl;
        std::cout << output_evaluated_by_CPU[i] << std::endl;
    }
}