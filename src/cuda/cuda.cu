#ifndef __CUDA_MAIN
#define __CUDA_MAIN
#include "../easypng/png.h"
#include "../util/png2arrays.h"
#include "kernels.cu"

void cuda_edge_algorithm(PNG * image) {
    png2arrays converter;
    converter.parse_png(image);

    int size = image->width() * image->height();
 
    //allocate the device input arrays
    float *r_dev;
    float *g_dev;
    float *b_dev;
 
    cudaMalloc((void**) &r_dev, size * sizeof(float));
    cudaMalloc((void**) &g_dev, size * sizeof(float));
    cudaMalloc((void**) &b_dev, size * sizeof(float));
 
    //allocate device output arrays
    float *grayscale_dev;
 
    cudaMalloc((void**) &grayscale_dev, size * sizeof(float));
 
    //copy input to device
    cudaMemcpy(r_dev, converter.r, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(g_dev, converter.g, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, converter.b, size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dim_grid_gray(image->width() / 16 + 1, image->height() / 16 + 1, 1);
    dim3 dim_block_gray(16, 16, 1);
    dim3 dim_grid_gauss(image->width()/OUTPUT_TILE_SIZE + 1,image->height()/OUTPUT_TILE_SIZE + 1, 1);
    dim3 dim_block_gauss(INPUT_TILE_SIZE, INPUT_TILE_SIZE, 1);

    grayscale_kernel<<<dim_block_gray, dim_grid_gray>>>(r_dev, g_dev, b_dev, grayscale_dev, image->width(), image->height());

    cudaFree(r_dev);
    cudaFree(g_dev);
    cudaFree(b_dev);

    float *gray_gauss_dev;
    cudaMalloc((void**) &gray_gauss_dev, size * sizeof(float));
    
    //allocate device variables for gradient calculations
    float *grad_dev;
    float *grad_x_dev;
    float *grad_y_dev;

    cudaMalloc((void**) &grad_dev, size * sizeof(float));
    cudaMalloc((void**) &grad_x_dev, size * sizeof(float));
    cudaMalloc((void**) &grad_y_dev, size * sizeof(float));

    
}

#endif //__CUDA_MAIN
