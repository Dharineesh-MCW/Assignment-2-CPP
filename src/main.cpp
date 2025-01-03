#include <iostream>
#include <fstream>
#include <string>
#include <nlohmann/json.hpp>
#include "Convolution2D.h"

using json = nlohmann::json;

void processLayer(const json& layer) {
    try{
    std::string type = layer["type"];

    if (type == "Conv2D") {
        
    } else if (type == "BatchNormalization") {
        
    } else if (type == "MaxPooling2D") {
        
    } else if (type == "Dropout") {
        
    } else if (type == "Flatten") {
        
    } else if (type == "Dense") {
        
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
    std::vector<std::vector<std::vector<float>>> vec3D(
        32, std::vector<std::vector<float>>(
                   32, std::vector<float>(3, 1.0f)));

    std::ifstream inputFile("O:/Assignment2/Project_Root/configs/json/final_configuration_model.json");
    if (!inputFile.is_open()) {
        std::cerr << "Failed to open the JSON file." << std::endl;
        return 1;
    }

    json layersJson;
    inputFile >> layersJson;
    inputFile.close();

    


    // Iterate through each layer and process it
    for (const auto& layer : layersJson["layers"]) {
        
    }

    return 0;
}
