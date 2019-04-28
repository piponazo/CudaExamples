#include <lib/utils.h>

#include <gflags/gflags.h>

#include <iostream>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime_api.h>

DECLARE_string(path);
DEFINE_string(path, "", "Path to image");

using namespace std;

namespace
{
const int CHANNELS = 3;
}

__global__ void colorToGreyscale(unsigned char *Pout, unsigned char *Pin, std::uint32_t width, std::uint32_t height)
{
    const std::uint32_t Col = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t Row = blockIdx.y * blockDim.y + threadIdx.y;

    if (Col < width && Row < height) {
        // get 1D coordinate for the grayscale image
        int greyOffset = Row * width + Col;
        int rgbOffset = greyOffset * CHANNELS;

        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset+1];
        unsigned char b = Pin[rgbOffset+2];

        Pout[greyOffset] = 0.21f * r + 0.71f*g + 0.07f*b;
    }
}

std::vector<unsigned char> convertImage(const std::vector<unsigned char>&input, std::uint32_t w, std::uint32_t h)
{
    const size_t rgbImageSize = input.size();
    const size_t greyImageSize = rgbImageSize / CHANNELS;
    unsigned char *d_in;
    unsigned char *d_out;

    cudaMalloc(&d_in, rgbImageSize);
    cudaMalloc(&d_out, greyImageSize);

    cudaMemcpy(d_in, input.data(), rgbImageSize, cudaMemcpyHostToDevice);
    dim3 dimGrid(std::ceil(w)/16., std::ceil(h/16.), 1);
    dim3 dimBlock(16, 16, 1);
    colorToGreyscale<<<dimGrid, dimBlock>>>(d_out, d_in, w, h);

    std::vector<unsigned char> greyImage(greyImageSize);
    cudaMemcpy(greyImage.data(), d_out, greyImageSize, cudaMemcpyDeviceToHost);

    return greyImage;
}

int main(int argc, char **argv)
{
    gflags::SetUsageMessage("appName -path=imagePath");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if (FLAGS_path.empty()) {
        cerr << gflags::ProgramUsage() << endl;
        return EXIT_FAILURE;
    }

    cout << "Image: " << FLAGS_path << endl;

    std::uint32_t width, height;
    std::vector<unsigned char> image = read_JPEG_file(FLAGS_path.c_str(), width, height);

    cout << "Image size in bytes: " << image.size() << endl;
    cout << "Image size pixels: " << width << " x " << height << endl;

    auto greyImge = convertImage(image, width, height);
    write_JPEG_file(greyImge, "prueba.jpg", width, height, 95);

    return EXIT_SUCCESS;
}
