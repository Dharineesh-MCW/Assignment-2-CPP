#include "Convolution2D.h"
#include <iostream>
#include <cmath>
#include <algorithm>

Convolution2D::Convolution2D(
    const std::vector<int>& inputShape,
    const std::vector<int>& outputShape,
    const std::vector<int>& kernelSize,
    const std::vector<int>& strides,
    const std::string& padding,
    const std::string& activation
)
    : inputShape(inputShape),
      outputShape(outputShape),
      kernelSize(kernelSize),
      strides(strides),
      padding(padding),
      activation(activation) {}

void Convolution2D::setInputData(const std::vector<std::vector<std::vector<float>>>& inputData) {
    this->inputData = inputData;
}

void Convolution2D::setKernelData(const std::vector<std::vector<std::vector<std::vector<float>>>>& kernelData) {
    this->kernelData = kernelData;
}

void Convolution2D::setBiasData(const std::vector<float>& biasData) {
    this->biasData = biasData;
}

void Convolution2D::performConvolution() {
    // Initialize output data
    outputData.resize(outputShape[0], std::vector<std::vector<float>>(outputShape[1], std::vector<float>(outputShape[2])));

    for (int h = 0; h < outputShape[0]; ++h) {
        for (int w = 0; w < outputShape[1]; ++w) {
            for (int f = 0; f < outputShape[2]; ++f) {
                float sum = 0.0f;
                for (int kh = 0; kh < kernelSize[0]; ++kh) {
                    for (int kw = 0; kw < kernelSize[1]; ++kw) {
                        for (int c = 0; c < inputShape[2]; ++c) {
                            int ih = h + kh - kernelSize[0] / 2;
                            int iw = w + kw - kernelSize[1] / 2;
                            if (padding == "same" && ih >= 0 && iw >= 0 && ih < inputShape[0] && iw < inputShape[1]) {
                                sum += inputData[ih][iw][c] * kernelData[f][kh][kw][c];
                            }
                        }
                    }
                }
                outputData[h][w][f] = sum + biasData[f];
            }
        }
    }
}

void Convolution2D::applyActivation() {
    if (activation == "relu") {
        for (auto& row : outputData) {
            for (auto& col : row) {
                for (auto& val : col) {
                    val = std::max(0.0f, val);
                }
            }
        }
    }
}

const std::vector<std::vector<std::vector<float>>>& Convolution2D::getOutputData() const {
    return outputData;
}
