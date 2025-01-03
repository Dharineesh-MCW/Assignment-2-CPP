#ifndef BATCHNORMALIZATION_H
#define BATCHNORMALIZATION_H

#include <vector>
#include <string>

class BatchNormalization {
public:
    // Constructor to initialize the class with weights file paths and shapes
    BatchNormalization(const std::vector<std::string>& weights_file_paths,
                       const std::vector<int>& input_shape,
                       const std::vector<int>& output_shape);

    // Method to perform batch normalization
    void normalize(const std::vector<std::vector<std::vector<float>>>& input, 
                   std::vector<std::vector<std::vector<float>>>& output);

private:
    std::vector<std::string> weights_file_paths;
    std::vector<int> input_shape;
    std::vector<int> output_shape;

    std::vector<float> gamma;
    std::vector<float> beta;
    std::vector<float> moving_mean;
    std::vector<float> moving_variance;

    // Method to load weights from files
    void loadWeights();

    // Method to apply batch normalization on the input
    void applyNormalization(const std::vector<std::vector<std::vector<float>>>& input,
                            std::vector<std::vector<std::vector<float>>>& output);
};

#endif // BATCHNORMALIZATION_H
