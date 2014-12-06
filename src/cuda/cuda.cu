#ifndef __CUDA_MAIN
#define __CUDA_MAIN
#include "../easypng/png.h"
#include "../util/png2arrays.h"
#include "kernels.cu"

void print_image(float * cuda_handle, string filename, int width, int height) {
    float * grayscale_host = (float *)malloc(width * height * sizeof(float));

    cudaMemcpy(grayscale_host, cuda_handle, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    png2arrays from_grayscale;

    from_grayscale.r = grayscale_host;
    from_grayscale.g = grayscale_host;
    from_grayscale.b = grayscale_host;
    from_grayscale.x_dim = width;
    from_grayscale.y_dim = height;

    PNG * grayscale = from_grayscale.from_arrays();
    grayscale->writeToFile(filename);

    delete grayscale;
    delete grayscale_host;
}

void cuda_edge_algorithm(PNG * image) {
    png2arrays converter;
    converter.parse_png(image);

    cudaError_t cuda_ret;

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

	cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) 
    {   
        printf("%s\n", cudaGetErrorString(cuda_ret));
    }
    cudaFree(r_dev);
    cudaFree(g_dev);
    cudaFree(b_dev);

    #ifdef TEST_MODE
        print_image(grayscale_dev, "grayscale.png", image->width(), image->height());
    #endif //TEST_MODE

    float *gray_gauss_dev;
    cudaMalloc((void**) &gray_gauss_dev, size * sizeof(float));

    gaussian_filter_kernel<<<dim_block_gauss, dim_grid_gauss>>>(grayscale_dev, gray_gauss_dev, image->width(), image->height());

	cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) 
    {   
        printf("%s\n", cudaGetErrorString(cuda_ret));
    }
    #ifdef TEST_MODE
        print_image(gray_gauss_dev, "gauss.png", image->width(), image->height());
    #endif //TEST_MODE

    //allocate device variables for gradient calculations
    float *grad_dev;
    float *grad_x_dev;
    float *grad_y_dev;

    cudaMalloc((void**) &grad_dev, size * sizeof(float));
    cudaMalloc((void**) &grad_x_dev, size * sizeof(float));
    cudaMalloc((void**) &grad_y_dev, size * sizeof(float));

    dim3 dim_grid_grad(image->width()/OUTPUT_TILE_SIZE + 1,image->height()/OUTPUT_TILE_SIZE + 1, 1);
    dim3 dim_block_grad(INPUT_TILE_SIZE_GRAD, INPUT_TILE_SIZE_GRAD, 1);
   
    gradient_calc_kernel<<<dim_block_grad, dim_grid_grad>>>(gray_gauss_dev, grad_x_dev, grad_y_dev, grad_dev, image->width(), image->height());

	cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) 
    {   
        printf("%s\n", cudaGetErrorString(cuda_ret));
    }
    #ifdef TEST_MODE
        print_image(grad_dev, "gradients.png", image->width(), image->height());
    #endif //TEST_MODE
    cudaFree(gray_gauss_dev);



    int * theta_dev;

    cudaMalloc((void**) &theta_dev, size * sizeof(int));
    //the directions are labeled counterclockwise from theta=0
    theta_calc_kernel<<<dim_block_gray, dim_grid_gray>>>(grad_x_dev, grad_y_dev, theta_dev, image->width(), image->height());

	cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) 
    {   
        printf("%s\n", cudaGetErrorString(cuda_ret));
    }
    cudaFree(grad_x_dev);
    cudaFree(grad_y_dev);

    dim3 dim_grid_trace(image->width()/16, image->height()/16, 1);
    dim3 dim_block_trace(18, 18, 1);

}

#endif //__CUDA_MAIN
