#include "Dense.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cmath>

// Constructor
Dense::Dense(const std::vector<int>& input_shape, const std::vector<int>& output_shape, 
             const std::string& activation, const std::string& weights_file, 
             const std::string& bias_file)
    : input_size_(input_shape[0]), output_size_(output_shape[0]), activation_(activation) {
    loadWeightsAndBias(weights_file, bias_file);
}

// Function to load weights and bias from files
void Dense::loadWeightsAndBias(const std::string& weights_file, const std::string& bias_file) {
    std::ifstream weightsStream(weights_file, std::ios::binary);
    std::ifstream biasStream(bias_file, std::ios::binary);
    
    if (!weightsStream.is_open() || !biasStream.is_open()) {
        throw std::runtime_error("Failed to open weights or bias files.");
    }

    // Load weights (input_size_ * output_size_ matrix)
    weights_.resize(input_size_ * output_size_);
    weightsStream.read(reinterpret_cast<char*>(weights_.data()), weights_.size() * sizeof(float));

    // Load bias (output_size_ vector)
    bias_.resize(output_size_);
    biasStream.read(reinterpret_cast<char*>(bias_.data()), bias_.size() * sizeof(float));

    weightsStream.close();
    biasStream.close();
}

// ReLU activation function
void Dense::applyReLU(std::vector<float>& output) {
    for (float& val : output) {
        val = std::max(0.0f, val); // ReLU: max(0, x)
    }
}

// Softmax activation function
void Dense::applySoftmax(std::vector<float>& output) {
    float sum_exp = 0.0f;
    for (const float& val : output) {
        sum_exp += std::exp(val);  // Calculate sum of exponentials
    }

    for (float& val : output) {
        val = std::exp(val) / sum_exp;  // Softmax formula: exp(x) / sum(exp(x))
    }
}

// Forward pass through the dense layer
void Dense::applyDense(const std::vector<float>& input, std::vector<float>& output) {
    if (input.size() != input_size_) {
        throw std::invalid_argument("Input size does not match the expected input size.");
    }

    // Initialize output vector with zeros
    output.resize(output_size_, 0.0f);

    // Perform matrix multiplication of input and weights + add bias
    for (int i = 0; i < output_size_; ++i) {
        for (int j = 0; j < input_size_; ++j) {
            output[i] += input[j] * weights_[i * input_size_ + j];
        }
        output[i] += bias_[i]; // Add bias
    }

    // Apply the appropriate activation function
    if (activation_ == "relu") {
        applyReLU(output);
    } else if (activation_ == "softmax") {
        applySoftmax(output);
    } else {
        std::cerr << "Unknown activation function: " << activation_ << "\n";
    }
}
