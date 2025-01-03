#include <iostream>
#include <fstream>
#include <string>
#include <nlohmann/json.hpp>
#include "Convolution2D.h"
#include "BatchNormalization.h"
#include "MaxPooling2D.h"
#include "Flatten.h"
#include "Dense.h"




using json = nlohmann::json;

void processLayer(const json& layer, std::vector<std::vector<std::vector<float>>>& input, std::vector<float> & oneD_Vector_Input) {
    try{
    std::string type = layer["type"];

    if (type == "Conv2D") {
        std::cout << "Performing Conv2D operation.\n";

        // Define layer parameters
        std::vector<int> inputShape =  layer["attributes"]["input_shape"];
        std::vector<int> outputShape = layer["attributes"]["output_shape"];
        std::vector<int> kernelSize = layer["attributes"]["kernel_size"];
        std::string kernelWeightPath = layer["weights_file_paths"][0];
        std::string biasWeightPath = layer["weights_file_paths"][1];
        std::vector<int> strides = {1, 1};
        std::string padding = layer["attributes"]["padding"];
        std::string activation = layer["attributes"]["activation"];

        // Initialize Conv2D layer
        Convolution2D convLayer(inputShape, outputShape, kernelSize, strides, padding, activation);

        // Load weights
        std::ifstream kernelFile(kernelWeightPath, std::ios::binary);
        if (!kernelFile.is_open()) {
            std::cerr << "Failed to open kernel file: " << kernelWeightPath << std::endl;
            return;
        }

        std::vector<std::vector<std::vector<std::vector<float>>>> kernelData(outputShape[2],
            std::vector<std::vector<std::vector<float>>>(kernelSize[0],
                std::vector<std::vector<float>>(kernelSize[1], std::vector<float>(inputShape[2]))));
        for (int f = 0; f < outputShape[2]; ++f) {
            for (int kh = 0; kh < kernelSize[0]; ++kh) {
                for (int kw = 0; kw < kernelSize[1]; ++kw) {
                    for (int c = 0; c < inputShape[2]; ++c) {
                        kernelFile.read(reinterpret_cast<char*>(&kernelData[f][kh][kw][c]), sizeof(float));
                    }
                }
            }
        }
        kernelFile.close();

        std::ifstream biasFile(biasWeightPath, std::ios::binary);
        if (!biasFile.is_open()) {
            std::cerr << "Failed to open bias file: " << biasWeightPath << std::endl;
            return;
        }

        std::vector<float> biasData(outputShape[2]);
        for (int f = 0; f < outputShape[2]; ++f) {
            biasFile.read(reinterpret_cast<char*>(&biasData[f]), sizeof(float));
        }
        biasFile.close();

        // Set weights in the layer
        convLayer.setKernelData(kernelData);
        convLayer.setBiasData(biasData);


        convLayer.setInputData(input);

        // Perform forward pass
        convLayer.performConvolution();
        convLayer.applyActivation();

        // Get and store the output
        std::vector<std::vector<std::vector<float>>> output = convLayer.getOutputData();
        input = output;  // Pass the output as input for the next layer (if applicable)
        size_t dim1 = input.size(); // First dimension
        size_t dim2 = input[0].size(); // Second dimension
        size_t dim3 = input[0][0].size(); // Third dimension

        // Print sizes
        std::cout << "Size of 3D vector:" << std::endl;
        std::cout << "Dimension 1: " << dim1 << std::endl;
        std::cout << "Dimension 2: " << dim2 << std::endl;
        std::cout << "Dimension 3: " << dim3 << std::endl;

       

        std::cout << "Conv2D operation completed successfully." << std::endl;
    }   
    else if (type == "BatchNormalization") {
        std::cout << "Performing Batch operation.-----------\n";

        std::vector<std::string> weights_file_paths = layer["weights_file_paths"];

        std::vector<int> input_shape =  layer["attributes"]["input_shape"];
        std::vector<int> output_shape = layer["attributes"]["output_shape"];

        // Create the BatchNormalization object using the provided data
        BatchNormalization batchNorm(weights_file_paths, input_shape, output_shape);

        // Create the input 3D vector (dummy data for illustration)
        int batch_size = input_shape[0];
        int height = input_shape[1];
        int width = input_shape[2];

        // Create the output 3D vector
        std::vector<std::vector<std::vector<float>>> output(output_shape[0], 
                                                        std::vector<std::vector<float>>(output_shape[1], 
                                                        std::vector<float>(output_shape[2], 0.0f)));

        // Normalize the input and get the output
        batchNorm.normalize(input, output);
        input = output;

        // Print the output to verify
        std::cout << "First element after batch normalization: " << output[0][0][0] << std::endl;

       
        std::cout << "Batch operation completed successfully. " << std::endl;
    } else if (type == "MaxPooling2D") {
        std::cout << "Performing MaxPooling2D operation.\n";

        // Define input configuration (this could come from JSON or another source)
        std::vector<int> inputShape =  layer["attributes"]["input_shape"];
        std::vector<int> outputShape = layer["attributes"]["output_shape"];
        std::vector<int> strides = layer["attributes"]["strides"];                 // [stride_height, stride_width]
        std::string padding = layer["attributes"]["padding"];                      // Padding type

        

        // Instantiate the MaxPooling2D layer
        MaxPooling2D maxPoolingLayer(inputShape, outputShape, strides, padding);

        // Output container (3D vector)
        std::vector<std::vector<std::vector<float>>> output;

        // Apply max pooling
        maxPoolingLayer.applyPooling(input, output);
        input = output;
        
        std::cout << "maxpooling done -----------" << "\n";

        
    } else if (type == "Dropout") {
        
    } else if (type == "Flatten") {

        std::cout << "Performing Flatten operation.\n";

    // Define input configuration (this could come from JSON or another source)
        std::vector<int> inputShape =  layer["attributes"]["input_shape"];
        std::vector<int> outputShape = layer["attributes"]["output_shape"];

        // Instantiate the Flatten layer
        Flatten flattenLayer(inputShape, outputShape);

        // Output container (1D vector)
        std::vector<float> output;

        // Apply flattening
        flattenLayer.applyFlatten(input, output);
        oneD_Vector_Input = output;
        std::cout << "\n In Flatten------ " << oneD_Vector_Input.size()  << "\n" ;

        // Display output shape and some values
        std::cout << "Output size: " << output.size() << "\n";
        std::cout << "First 10 elements of the output: ";
        for (int i = 0; i < std::min(10, static_cast<int>(output.size())); ++i) {
            std::cout << output[i] << " ";
        }
        std::cout << "\n----------Flatten Done----------------------------------\n";
        
    } else if (type == "Dense") {
        std::cout << "Performing Dense layer operation.\n";

        // Define input configuration (these would come from the previous layer output, typically flattened)
        std::vector<int> inputShape =  layer["attributes"]["input_shape"];
        std::vector<int> outputShape = layer["attributes"]["output_shape"];
        std::string activation = layer["attributes"]["activation"];  // Choose between "relu" or "softmax"
        std::vector<std::string> dense_weight_path = layer["weights_file_paths"];
        

        // Example: the 1D input vector (this is what you get from the Flatten layer)
        // std::vector<float> input(inputShape[0], 1.0f);  // Fill with dummy data, replace with actual values

        // Instantiate the Dense layer with the chosen activation function
        Dense denseLayer(inputShape, outputShape, activation, dense_weight_path[0],dense_weight_path[1]);

        // Output container (1D vector)
        std::vector<float> output;

        // Apply Dense layer
        std::cout << oneD_Vector_Input.size() << " " << inputShape[0] << "\n";
        denseLayer.applyDense(oneD_Vector_Input, output);
        oneD_Vector_Input = output;

        // Display output shape and first few values
        float val = -1.0;
        int ind = -1;
        std::cout << "Output size: " << output.size() << "\n";
        if(output.size() == 10){
        for (int i = 0; i < output.size(); ++i) {
            std::cout << output[i] << " ";
            if(output[i] > val){
                val = output[i];
                ind = i;
            }
        }
        std::cout <<"\n index of the class : " <<  ind << "\n";
        }
        
    } else {
        std::cerr << "Unknown layer type: " << type << std::endl;
    }
    }
    catch(const std::exception& e) {
        std::cerr << "Error processing layer: " << e.what() << std::endl;
    }
}

int main() {
    // Load JSON file
    std::vector<std::vector<std::vector<float>>> input(
        32, std::vector<std::vector<float>>(
                   32, std::vector<float>(3, 1.0f)));
    std::vector<float> oneD_Vector_Input;

    std::ifstream inputFile("O:/Assignment2/Project_Root/configs/json/updated_configuration_model.json");
    if (!inputFile.is_open()) {
        std::cerr << "Failed to open the JSON file." << std::endl;
        return 1;
    }

    json layersJson;
    inputFile >> layersJson;
    inputFile.close();

    


    // Iterate through each layer and process it
    int c = 0;
    for (const auto& layer : layersJson["layers"]) {
        processLayer(layer, input, oneD_Vector_Input);
    }

    return 0;
}
