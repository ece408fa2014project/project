#ifndef __FACE_DETECTION_MAIN
#define __FACE_DETECTION_MAIN
#include "../easypng/png.h"
#include "../util/png2arrays.h"
#include "kernels.cu"

void do_face_detection_cuda(float * r, float * g, float * b, float * out_r, float * out_g,  float * out_b, int width, int height) {
    
    int size = width * height;
    //allocate the device input arrays
    float *r_dev;
    float *g_dev;
    float *b_dev;
    
    cudaMalloc((void**) &r_dev, size * sizeof(float));
    cudaMalloc((void**) &g_dev, size * sizeof(float));
    cudaMalloc((void**) &b_dev, size * sizeof(float));
    
    //allocate device output arrays
    float *grayscale_dev_1;
    
    cudaMalloc((void**) &grayscale_dev_1, size * sizeof(float));
    
    float *grayscale_dev_2;
    
    cudaMalloc((void**) &grayscale_dev_2, size * sizeof(float));
    
    float *grayscale_dev_1_out;
    
    cudaMalloc((void**) &grayscale_dev_1_out, size * sizeof(float));
    
    float *grayscale_dev_2_out;
    
    cudaMalloc((void**) &grayscale_dev_2_out, size * sizeof(float));
    
    float *grayscale_dev_3_out;
    
    cudaMalloc((void**) &grayscale_dev_3_out, size * sizeof(float));
    
    float *grayscale_dev_4_out;
    
    cudaMalloc((void**) &grayscale_dev_4_out, size * sizeof(float));
    
    
    //copy input to device
    cudaMemcpy(r_dev, r, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(g_dev, g, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, b, size * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 dim_grid_gray(width / 16 + 1, height / 16 + 1, 1);
    dim3 dim_block_gray(16, 16, 1);
    
    skin_detection<<<dim_block_gray, dim_grid_gray>>>(r_dev, g_dev, b_dev, grayscale_dev_1, grayscale_dev_2, width, height);
    
    clean_up<<<dim_block_gray, dim_grid_gray>>>(grayscale_dev_1, grayscale_dev_2, grayscale_dev_3_out, grayscale_dev_4_out, width, height);
    
    quantization<<<dim_block_gray, dim_grid_gray>>>(grayscale_dev_3_out, grayscale_dev_4_out, grayscale_dev_1_out, grayscale_dev_2_out, width, height);
    
    cudaFree(r_dev);
    cudaFree(g_dev);
    cudaFree(b_dev);
    
    float * grayscale_host_skin = (float *)malloc(size * sizeof(float));
    cudaMemcpy(grayscale_host_skin, grayscale_dev_1_out, size * sizeof(float), cudaMemcpyDeviceToHost);
    float * grayscale_host_hair = (float *)malloc(size * sizeof(float));
    cudaMemcpy(grayscale_host_hair, grayscale_dev_2_out, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    face_detection(grayscale_host_skin, grayscale_host_hair, out_r, out_g, out_b, width, height);
    
    cudaFree(grayscale_dev_1);
    cudaFree(grayscale_dev_2);
    cudaFree(grayscale_dev_1_out);
    cudaFree(grayscale_dev_2_out);
    cudaFree(grayscale_dev_3_out);
    cudaFree(grayscale_dev_4_out);
}

#endif //__CUDA_MAIN
