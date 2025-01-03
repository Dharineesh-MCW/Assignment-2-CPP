#ifndef CONVOLUTION2D_H
#define CONVOLUTION2D_H

#include <vector>
#include <string>

class Convolution2D {
public:
    Convolution2D(
        const std::vector<int>& inputShape,
        const std::vector<int>& outputShape,
        const std::vector<int>& kernelSize,
        const std::vector<int>& strides,
        const std::string& padding,
        const std::string& activation
    );

    void setInputData(const std::vector<std::vector<std::vector<float>>>& inputData);
    void setKernelData(const std::vector<std::vector<std::vector<std::vector<float>>>>& kernelData);
    void setBiasData(const std::vector<float>& biasData);

    void performConvolution();
    void applyActivation();

    const std::vector<std::vector<std::vector<float>>>& getOutputData() const;

private:
    std::vector<int> inputShape; // Input shape (height, width, channels)
    std::vector<int> outputShape; // Output shape (height, width, filters)
    std::vector<int> kernelSize; // Kernel size (height, width)
    std::vector<int> strides;    // Stride (height, width)
    std::string padding;
    std::string activation;

    // Data storage
    std::vector<std::vector<std::vector<float>>> inputData;  // 3D input vector
    std::vector<std::vector<std::vector<std::vector<float>>>> kernelData;  // 4D kernel
    std::vector<float> biasData;  // Bias for each filter
    std::vector<std::vector<std::vector<float>>> outputData;  // 3D output vector
};

#endif // CONVOLUTION2D_H
