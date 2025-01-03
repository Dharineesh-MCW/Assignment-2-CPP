#include "MaxPooling2D.h"
#include <algorithm>
#include <iostream>
#include <limits>

MaxPooling2D::MaxPooling2D(const std::vector<int>& input_shape,
                           const std::vector<int>& output_shape,
                           const std::vector<int>& strides,
                           const std::string& padding_type)
    : input_height_(input_shape[0]), input_width_(input_shape[1]), input_depth_(input_shape[2]),
      stride_height_(strides[0]), stride_width_(strides[1]), padding_type_(padding_type) {
}

void MaxPooling2D::apply_padding(std::vector<std::vector<std::vector<float>>>& input) {
    // Add padding logic if 'valid' padding is specified.
    if (padding_type_ == "valid") {
        return; // No padding needed for "valid"
    }
    // Additional padding schemes like 'same' can be implemented here.
}

void MaxPooling2D::applyPooling(const std::vector<std::vector<std::vector<float>>>& input, 
                                std::vector<std::vector<std::vector<float>>>& output) {
    // Apply padding if necessary
    std::vector<std::vector<std::vector<float>>> padded_input = input;
    apply_padding(padded_input);

    // Calculate output dimensions
    int output_height = (input_height_ - 1) / stride_height_ + 1;
    int output_width = (input_width_ - 1) / stride_width_ + 1;

    output.resize(output_height, std::vector<std::vector<float>>(
        output_width, std::vector<float>(input_depth_, 0.0f)
    ));

    // Perform MaxPooling
    for (int depth = 0; depth < input_depth_; ++depth) {
        for (int i = 0; i < output_height; ++i) {
            for (int j = 0; j < output_width; ++j) {
                float max_val = -std::numeric_limits<float>::infinity();
                for (int m = 0; m < stride_height_; ++m) {
                    for (int n = 0; n < stride_width_; ++n) {
                        int input_i = i * stride_height_ + m;
                        int input_j = j * stride_width_ + n;
                        if (input_i < input_height_ && input_j < input_width_) {
                            max_val = std::max(max_val, padded_input[input_i][input_j][depth]);
                        }
                    }
                }
                output[i][j][depth] = max_val;
            }
        }
    }
}
