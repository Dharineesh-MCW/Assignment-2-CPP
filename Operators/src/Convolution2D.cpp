#include "Convolution2D.h"

Convolution2D::Convolution2D(const nlohmann::json& layerConfig) {
    layerName = layerConfig["layer_name"];
    inputFilePath = layerConfig["input_file_path"];
    outputFilePath = layerConfig["output_file_path"];
    weightsFilePaths = layerConfig["weights_file_paths"].get<std::vector<std::string>>();
    parseAttributes(layerConfig["attributes"]);
}

void Convolution2D::parseAttributes(const nlohmann::json& attributes) {
    inputShape = {32, 32, 3}; // Extracted as an example
    outputShape = {32, 32, 64}; // Extracted as an example
    kernelSize = attributes["kernel_size"].get<std::vector<int>>();
    strides = attributes["strides"].get<std::vector<int>>();
    padding = attributes["padding"];
    activation = attributes["activation"];
}

void Convolution2D::loadWeights() {
    // Implement weight loading logic here
    std::cout << "Loading weights from: " << weightsFilePaths[0] << " and " << weightsFilePaths[1] << std::endl;
}

void Convolution2D::processInput() {
    // Implement input processing logic here
    std::cout << "Processing input from: " << inputFilePath << std::endl;
    std::cout << "Writing output to: " << outputFilePath << std::endl;
}

void Convolution2D::execute() {
    std::cout << "Executing Convolution2D Layer: " << layerName << std::endl;
    loadWeights();
    processInput();
    std::cout << "Execution completed for layer: " << layerName << std::endl;
}
