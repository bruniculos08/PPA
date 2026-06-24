#include "perceptron.cuh"

void NeuralNetworkHost::printDenseNetwork(network &model)
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

void NeuralNetworkHost::trainDenseNetwork(network model, int data_size, double **data)
{
    for(size_t layer_index = 0; layer_index < model.layers_num; layer_index++)
    {
        layer actual_layer = model.layer_vector[layer_index];
        for(size_t i = 0; i < actual_layer.size; i++)
        {
            for(size_t j = 0; j < actual_layer.neurons[i].input_size; j++)
            {   
                double cost_w = NeuralNetworkHost::costDenseNetwork(model, data_size, data);
                
                ((actual_layer.neurons[i]).weights[j]) += EPSILON;
                double cost_w_epsilon = NeuralNetworkHost::costDenseNetwork(model, data_size, data);
                ((actual_layer.neurons[i]).weights[j]) -= EPSILON;

                ((actual_layer.neurons[i]).weights[j]) -= (cost_w_epsilon - cost_w) * EPSILON;
            }
            double cost_w = NeuralNetworkHost::costDenseNetwork(model, data_size, data);

            *((actual_layer.neurons[i]).b) += EPSILON;
            double cost_w_epsilon = NeuralNetworkHost::costDenseNetwork(model, data_size, data);
            *((actual_layer.neurons[i]).b) -= EPSILON;

            *((actual_layer.neurons[i]).b) -= (cost_w_epsilon - cost_w) * EPSILON;
        }
    }
}

void CudaManagementByHost::trainDenseNetworkUsingGPU(network *device_model_address, network *host_model_with_device_weights, int data_size, double **host_data)
{
    for(size_t layer_index = 0; layer_index < host_model_with_device_weights->layers_num; layer_index++)
    {
        layer &actual_layer = host_model_with_device_weights->layer_vector[layer_index];
        for(size_t i = 0; i < actual_layer.size; i++)
        {
            for(size_t j = 0; j < actual_layer.neurons[i].input_size; j++)
            {
                double *weight = ((double *) CudaManagementByHost::copyData((actual_layer.neurons[i]).weights + j, sizeof(double), cudaMemcpyDeviceToHost));

                double cost_w = CudaManagementByHost::costDenseNetworkUsingGPU(device_model_address, host_model_with_device_weights, data_size, host_data);
                *weight += EPSILON;
                cudaMemcpy((actual_layer.neurons[i]).weights + j, weight, sizeof(double), cudaMemcpyHostToDevice);
                double cost_w_epsilon = CudaManagementByHost::costDenseNetworkUsingGPU(device_model_address, host_model_with_device_weights, data_size, host_data);
                *weight -= EPSILON;
                cudaMemcpy((actual_layer.neurons[i]).weights + j, weight, sizeof(double), cudaMemcpyHostToDevice);

                *weight -= (cost_w_epsilon - cost_w) * EPSILON;
                cudaMemcpy((actual_layer.neurons[i]).weights + j, weight, sizeof(double), cudaMemcpyHostToDevice);
            }

            double *bias = ((double *) CudaManagementByHost::copyData((actual_layer.neurons[i]).b, sizeof(double), cudaMemcpyDeviceToHost));

            double cost_w = CudaManagementByHost::costDenseNetworkUsingGPU(device_model_address, host_model_with_device_weights, data_size, host_data);
            *bias += EPSILON;
            cudaMemcpy((actual_layer.neurons[i]).b, bias, sizeof(double), cudaMemcpyHostToDevice);
            double cost_w_epsilon = CudaManagementByHost::costDenseNetworkUsingGPU(device_model_address, host_model_with_device_weights, data_size, host_data);
            *bias -= EPSILON;
            cudaMemcpy((actual_layer.neurons[i]).b, bias, sizeof(double), cudaMemcpyHostToDevice);

            *bias -= (cost_w_epsilon - cost_w) * EPSILON;
            cudaMemcpy((actual_layer.neurons[i]).b, bias, sizeof(double), cudaMemcpyHostToDevice);
        }
    }
}

double NeuralNetworkHost::costDenseNetwork(network model, size_t data_size, double **data)
{
    size_t input_size = model.layer_vector[0].size;
    size_t output_size = model.layer_vector[model.layers_num - 1].size;
    double carry = 0.0;
    for(size_t i = 0; i < data_size; i++)
    {
        for(size_t j = 0; j < output_size; j++)
        {
            if(COST_FUNCTION == 0)
            {
                carry += fabs((data[i][input_size + j] - (NeuralNetworkHost::evaluateDenseInput(model, data[i]))[j])) / ((double) data_size);
            }
            else
            {
                carry += pow((data[i][input_size + j] - (NeuralNetworkHost::evaluateDenseInput(model, data[i]))[j]), 2.0) / ((double) data_size);
            }
        }
    }
    return carry;
}

double CudaManagementByHost::costDenseNetworkUsingGPU(network *device_model_address, network *host_model_with_device_weights, size_t data_size, double **host_data)
{
    size_t input_size = host_model_with_device_weights->layer_vector[0].size;
    size_t output_size = host_model_with_device_weights->layer_vector[host_model_with_device_weights->layers_num - 1].size;
    double carry = 0.0;
    for(size_t i = 0; i < data_size; i++)
    {
        double *device_data = (double *) CudaManagementByHost::copyData(host_data[i], input_size * sizeof(double), cudaMemcpyHostToDevice);
        double *evaluation = CudaManagementByHost::evaluateDenseInputUsingGPU(device_model_address, host_model_with_device_weights, device_data);
        for(size_t j = 0; j < output_size; j++)
        {
            if(COST_FUNCTION == 0)
            {
                carry += fabs((host_data[i][input_size + j] - (evaluation)[j])) / ((double) data_size);
            }
            else
            {
                carry += pow((host_data[i][input_size + j] - (evaluation)[j]), 2.0) / ((double) data_size);
            }
        }
    }
    return carry;
}

double NeuralNetworkHost::dotProduct(size_t dim, double *v1, double *v2)
{
    double carry = 0.0;
    for(size_t i = 0; i < dim; i++)
        carry += (v1[i] * v2[i]);
    return carry;
}

__device__ double NeuralNetworkDevice::dotProduct(size_t dim, double *v1, double *v2)
{
    double carry = 0.0;
    for(size_t i = 0; i < dim; i++)
        carry += (v1[i] * v2[i]);
    return carry;
}

double *NeuralNetworkHost::evaluateDenseInput(network model, double *input)
{   
    layer *actual_layer;
    double *actual_layer_output;
    double *last_layer_output;
    actual_layer_output = NULL;
    last_layer_output = input;
    int last_layer_size = model.layer_vector[0].neurons[0].input_size;
    for(size_t i = 0; i < (size_t) model.layers_num; i++)
    {
        actual_layer = &model.layer_vector[i];
        actual_layer_output = (double *) malloc(actual_layer->size * sizeof(double));
        for(size_t j = 0; j < (size_t) actual_layer->size; j++)
        {
            if(model.activation_function != NULL)
                actual_layer_output[j] = (*model.activation_function)(NeuralNetworkHost::dotProduct(last_layer_size, actual_layer->neurons[j].weights, last_layer_output)
                    + (*(actual_layer->neurons[j].b)));
            else 
                actual_layer_output[j] = NeuralNetworkHost::dotProduct(last_layer_size, actual_layer->neurons[j].weights, last_layer_output) + (*((actual_layer->neurons[j]).b));
        }
        if(last_layer_output != input) 
            free(last_layer_output);
        last_layer_output = actual_layer_output;
        last_layer_size = actual_layer->size;
    }
    return last_layer_output;
}

__global__ void NeuralNetworkDevice::evaluateLayerOutput(network *model, size_t layer_index, double *input, double *output)
{
    size_t index = threadIdx.y;
    layer *l = &model->layer_vector[0];
    perceptron *neuron = &l->neurons[index];
    if(model->activation_function != NULL)
        output[index] = (*model->activation_function)(NeuralNetworkDevice::dotProduct(neuron->input_size, neuron->weights, input) + (*(neuron->b)));
    else
        output[index] = (NeuralNetworkDevice::dotProduct(neuron->input_size, neuron->weights, input) + (*(neuron->b)));
}

double *CudaManagementByHost::evaluateDenseInputUsingGPU(network *device_model, network *host_model_with_device_weights, double *input)
{
    double *layer_input = input;
    double *layer_output;
    for (size_t i = 0; i < host_model_with_device_weights->layers_num; i++)
    {
        cudaMalloc(&layer_output, host_model_with_device_weights->layer_vector[i].size * sizeof(double));
        dim3 grid_dim(1);
        dim3 block_dim(1, host_model_with_device_weights->layer_vector[i].size);
        NeuralNetworkDevice::evaluateLayerOutput<<<grid_dim,block_dim>>>(device_model, i, layer_input, layer_output);
        if (layer_input != input) cudaFree(layer_input);
        layer_input = layer_output;
    }
    cudaDeviceSynchronize();
    size_t output_size = host_model_with_device_weights->layer_vector[host_model_with_device_weights->layers_num - 1].size;
    double *result = (double *) CudaManagementByHost::copyData(layer_output, output_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(layer_output);
    return result;
}

network *CudaManagementByHost::copyNetworkToGPU(network model)
{
    for (size_t i = 0; i < model.layers_num; i++)
    {
        layer *l = &model.layer_vector[i];
        for (size_t j = 0; j < l->size; j++)
        {
            perceptron *p = &l->neurons[i];
            p->weights = (double *) CudaManagementByHost::copyData(p->weights, p->input_size * sizeof(double), cudaMemcpyHostToDevice);
            p->b = (double *) CudaManagementByHost::copyData(p->b, sizeof(double), cudaMemcpyHostToDevice);
        }
        l->neurons = (perceptron *) CudaManagementByHost::copyData(l->neurons, l->size * sizeof(perceptron), cudaMemcpyHostToDevice);
    }
    model.layer_vector = (layer *) CudaManagementByHost::copyData(model.layer_vector, model.layers_num * sizeof(layer), cudaMemcpyHostToDevice);
    model.activation_function = NULL;
    network *model_copy = (network *) CudaManagementByHost::copyData(&model, sizeof(network), cudaMemcpyHostToDevice);
    return model_copy;
}

network *CudaManagementByHost::getWeightsFromGPU(network *device_model)
{
    network *host_model = ((network *) CudaManagementByHost::copyData(device_model, sizeof(network), cudaMemcpyDeviceToHost));
    host_model->layer_vector = (layer *) CudaManagementByHost::copyData(host_model->layer_vector, host_model->layers_num * sizeof(layer), cudaMemcpyDeviceToHost);
    for(size_t layer_index = 0; layer_index < host_model->layers_num; layer_index++)
    {
        host_model->layer_vector[layer_index].neurons = (perceptron *) CudaManagementByHost::copyData(
            host_model->layer_vector[layer_index].neurons,
            host_model->layer_vector[layer_index].size * sizeof(perceptron),
            cudaMemcpyDeviceToHost
        );
    }
    return host_model;
}

network NeuralNetworkHost::genUniformDenseNetwork(size_t layers_num, size_t layer_size, size_t input_size, size_t output_size, double(*activation_function)(double))
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
            l->neurons[i].b = (double *) malloc(sizeof(double));
            *(l->neurons[i].b) = 0.0;
            l->neurons[i].input_size = (i >= 1) ? model.layer_vector[i - 1].size : input_size;
            l->neurons[i].weights = (double *) malloc(l->neurons[i].input_size * sizeof(double));
            memset(l->neurons[i].weights, 0, l->neurons[i].input_size * sizeof(double));
        }
    }
    model.activation_function = activation_function;
    return model;
}

network NeuralNetworkHost::genDenseNetwork(size_t layers_num, size_t *layer_size, size_t input_size, size_t output_size, double(*activation_function)(double))
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
            l->neurons[i].b = (double *) malloc(sizeof(double));
            *(l->neurons[i].b) = 1.0;
            l->neurons[j].input_size = (i >= 1) ? model.layer_vector[i - 1].size : input_size;
            l->neurons[j].weights = (double *) malloc(l->neurons[j].input_size * sizeof(double));
            for (size_t weight_index = 0; weight_index < l->neurons[i].input_size; weight_index++)
                l->neurons[i].weights[weight_index] = 1.0;
        }
    }
    model.activation_function = activation_function;
    return model;
}

void NeuralNetworkHost::validateDenseNeuralNetwork(network model, double **data, size_t data_size, size_t input_size, size_t output_size)
{
    printf("evaluating validation dataset:\n");
    for(size_t i = 0; i < data_size; i++)
    {
        double *result = NeuralNetworkHost::evaluateDenseInput(model, data[i]);
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
    cudaDeviceReset();
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

    double input_test[] = {2, 1};
    size_t layers_num = 1;
    size_t layers_size[] = {(size_t) output_size};
    double **validation = (double **) malloc(sizeof(double *));
    validation[0] = (double *) malloc(2 * sizeof(double));
    validation[0][0] = 2;
    validation[0][1] = 1;

    network model = NeuralNetworkHost::genDenseNetwork(layers_num, layers_size, input_size, output_size, NULL);

    std::cout << "Starting CPU tests" << std::endl;
    auto beggining = std::chrono::steady_clock::now();

    for(size_t i = 0; i < (size_t) TRAINING_TIMES; i++) NeuralNetworkHost::trainDenseNetwork(model, (size_t) data_size, data);
    double *host_output = NeuralNetworkHost::evaluateDenseInput(model, input_test);

    auto ending = std::chrono::steady_clock::now();
    auto diff = ending - beggining;
    std::cout << "Ending CPU tests, time elapsed = " << std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() << std::endl;

    network m = NeuralNetworkHost::genDenseNetwork(layers_num, layers_size, input_size, output_size, NULL);
    network *device_model = CudaManagementByHost::copyNetworkToGPU(m);
    network *host_model_with_device_weights = CudaManagementByHost::getWeightsFromGPU(device_model);
    double *device_input = (double *) CudaManagementByHost::copyData(input_test, input_size * sizeof(double), cudaMemcpyHostToDevice);

    std::cout << "Starting GPU tests" << std::endl;
    beggining = std::chrono::steady_clock::now();

    for(size_t i = 0; i < (size_t) TRAINING_TIMES; i++) CudaManagementByHost::trainDenseNetworkUsingGPU(device_model, host_model_with_device_weights, (size_t) data_size, data);
    double *device_output = CudaManagementByHost::evaluateDenseInputUsingGPU(device_model, host_model_with_device_weights, device_input);

    ending = std::chrono::steady_clock::now();
    diff = ending - beggining;
    std::cout << "Ending GPU tests, time elapsed = " << std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() << std::endl;

    std::cout << "Evaluation results:" << std::endl;
    for (size_t i = 0; i < output_size; i++)
    {
        std::cout << "device_output[" << i << "] = " << device_output[i] << std::endl;
        std::cout << "host_output [" << i << "] = " << host_output[i] << std::endl;
    }
    return 0;
}