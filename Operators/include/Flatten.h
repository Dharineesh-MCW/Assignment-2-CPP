#ifndef FLATTEN_H
#define FLATTEN_H

#include <vector>

class Flatten {
public:
    Flatten(const std::vector<int>& input_shape, const std::vector<int>& output_shape);

    // Flatten 3D input tensor into 1D vector
    void applyFlatten(const std::vector<std::vector<std::vector<float>>>& input, 
                      std::vector<float>& output);

private:
    int input_height_;
    int input_width_;
    int input_depth_;
    int output_size_;
};

#endif // FLATTEN_H
