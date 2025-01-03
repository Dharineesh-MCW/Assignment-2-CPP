#ifndef MAX_POOLING2D_H
#define MAX_POOLING2D_H

#include <vector>
#include <string>

class MaxPooling2D {
public:
    MaxPooling2D(const std::vector<int>& input_shape,
                 const std::vector<int>& output_shape,
                 const std::vector<int>& strides,
                 const std::string& padding_type);

    // Apply max pooling to 3D input
    void applyPooling(const std::vector<std::vector<std::vector<float>>>& input, 
                      std::vector<std::vector<std::vector<float>>>& output);

private:
    int input_height_;
    int input_width_;
    int input_depth_;
    int stride_height_;
    int stride_width_;
    std::string padding_type_;

    void apply_padding(std::vector<std::vector<std::vector<float>>>& input);
};

#endif // MAX_POOLING2D_H
