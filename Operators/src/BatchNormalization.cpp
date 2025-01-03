#include "BatchNormalization.h"
#include <fstream>
#include <iostream>
#include <cmath>

BatchNormalization::BatchNormalization(const std::vector<std::string>& weights_file_paths,
                                       const std::vector<int>& input_shape,
                                       const std::vector<int>& output_shape)
    : weights_file_paths(weights_file_paths),
      input_shape(input_shape),
      output_shape(output_shape) {
    loadWeights();
}

void BatchNormalization::loadWeights() {
    // Load gamma, beta, moving mean, and moving variance from their respective files
    for (const auto& weight_path : weights_file_paths) {
        std::ifstream file(weight_path, std::ios::binary);
        if (!file) {
            std::cerr << "Error opening weight file: " << weight_path << std::endl;
            exit(1);
        }

        // Assuming each weight file contains one float value per entry
        std::vector<float> temp;
        float value;
        while (file.read(reinterpret_cast<char*>(&value), sizeof(float))) {
            temp.push_back(value);
        }

        if (weight_path.find("gamma") != std::string::npos) {
            gamma = temp;
        } else if (weight_path.find("beta") != std::string::npos) {
            beta = temp;
        } else if (weight_path.find("moving_mean") != std::string::npos) {
            moving_mean = temp;
        } else if (weight_path.find("moving_variance") != std::string::npos) {
            moving_variance = temp;
        }
    }
}

void BatchNormalization::normalize(const std::vector<std::vector<std::vector<float>>>& input,
                                   std::vector<std::vector<std::vector<float>>>& output) {
    applyNormalization(input, output);
}

void BatchNormalization::applyNormalization(const std::vector<std::vector<std::vector<float>>>& input,
                                            std::vector<std::vector<std::vector<float>>>& output) {
    int batch_size = input_shape[0];
    int height = input_shape[1];
    int width = input_shape[2];

    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                for (int c = 0; c < input_shape[2]; ++c) {
                    float normalized_value = (input[b][h][w] - moving_mean[c]) /
                                             std::sqrt(moving_variance[c] + 1e-5);
                    output[b][h][w] = gamma[c] * normalized_value + beta[c];
                }
            }
        }
    }
}
