#include "Flatten.h"
#include <iostream>

Flatten::Flatten(const std::vector<int>& input_shape, const std::vector<int>& output_shape)
    : input_height_(input_shape[0]), input_width_(input_shape[1]), input_depth_(input_shape[2]),
      output_size_(output_shape[0]) {
}

void Flatten::applyFlatten(const std::vector<std::vector<std::vector<float>>>& input, 
                           std::vector<float>& output) {
    // Resize the output vector to match the flattened size
    output.resize(output_size_);

    int index = 0;
    // Flatten the 3D input tensor into 1D vector
    for (int h = 0; h < input_height_; ++h) {
        for (int w = 0; w < input_width_; ++w) {
            for (int d = 0; d < input_depth_; ++d) {
                output[index++] = input[h][w][d];
            }
        }
    }
}
