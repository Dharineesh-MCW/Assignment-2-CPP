#ifndef CONVOLUTION2D_H
#define CONVOLUTION2D_H

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

class Convolution2D {
public:
    Convolution2D(const nlohmann::json& layerConfig);
    void execute();

private:
    std::string layerName;
    std::string inputFilePath;
    std::string outputFilePath;
    std::vector<std::string> weightsFilePaths;
    std::vector<int> kernelSize;
    std::vector<int> strides;
    std::string padding;
    std::string activation;
    std::vector<int> inputShape;
    std::vector<int> outputShape;

    void parseAttributes(const nlohmann::json& attributes);
    void loadWeights();
    void processInput();
};

#endif // CONVOLUTION2D_H
