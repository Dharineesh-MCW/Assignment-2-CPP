#ifndef DENSE_H
#define DENSE_H

#include <vector>
#include <string>

class Dense {
public:
    Dense(const std::vector<int>& input_shape, const std::vector<int>& output_shape, 
          const std::string& activation, const std::string& weights_file, 
          const std::string& bias_file);

    // Forward pass function
    void applyDense(const std::vector<float>& input, std::vector<float>& output);

private:
    int input_size_;
    int output_size_;
    std::string activation_;
    std::vector<float> weights_; // Flattened weights matrix
    std::vector<float> bias_;    // Bias vector

    // Activation functions
    void applyReLU(std::vector<float>& output);
    void applySoftmax(std::vector<float>& output);
    
    // Function to load weights and bias from files
    void loadWeightsAndBias(const std::string& weights_file, const std::string& bias_file);
};

#endif // DENSE_H
