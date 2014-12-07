#include "../easypng/png.h"
#include "../util/png2arrays.h"
#include "kernels.cu"

void do_edge_detection_cuda(float * r, float * g, float * b, float * out, float * prevFrame1, float * prevFrame2, float * prevFrame3, int width, int height) {
    cudaError_t cuda_ret;

    int size = width * height;

    float *r_dev, *g_dev, *b_dev;

    cudaMalloc((void**) &r_dev, size * sizeof(float));
    cudaMalloc((void**) &g_dev, size * sizeof(float));
    cudaMalloc((void**) &b_dev, size * sizeof(float));

    //allocate device output arrays
    float *grayscale_dev;

    cudaMalloc((void**) &grayscale_dev, size * sizeof(float));

    //copy input to device
    cudaMemcpy(r_dev, r, size * sizeof(float), cudaMemcpyHostToDevice);

    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess)
      {
        printf("%s\n", cudaGetErrorString(cuda_ret));
      }
    cudaMemcpy(g_dev, g, size * sizeof(float), cudaMemcpyHostToDevice);

    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess)
      {
        printf("%s\n", cudaGetErrorString(cuda_ret));
      }
    cudaMemcpy(b_dev, b, size * sizeof(float), cudaMemcpyHostToDevice);

    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess)
      {
          printf("%s\n", cudaGetErrorString(cuda_ret));
      }

    dim3 dim_grid_gray(width / 16 + 1, height / 16 + 1, 1);
    dim3 dim_block_gray(16, 16, 1);

    grayscale_kernel<<<dim_grid_gray, dim_block_gray>>>(r_dev, g_dev, b_dev, grayscale_dev, width, height);

	cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess)
    {
        printf("%s\n", cudaGetErrorString(cuda_ret));
    }
    cudaFree(r_dev);
    cudaFree(g_dev);
    cudaFree(b_dev);

    dim3 dim_grid_gauss(width/12 + 1,height/12 + 1, 1);
    dim3 dim_block_gauss(16, 16, 1);

    float *gray_gauss_dev;
    cudaMalloc((void**) &gray_gauss_dev, size * sizeof(float));
    gaussian_filter_kernel<<<dim_grid_gauss, dim_block_gauss>>>(grayscale_dev, gray_gauss_dev, width, height);

	cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess)
    {
        printf("%s\n", cudaGetErrorString(cuda_ret));
    }

    cudaFree(grayscale_dev);

    float *grad_dev;

    cudaMalloc((void**) &grad_dev, size * sizeof(float));

    dim3 dim_grid_grad(width/OUTPUT_TILE_SIZE + 1,height/OUTPUT_TILE_SIZE + 1, 1);
    dim3 dim_block_grad(INPUT_TILE_SIZE_GRAD, INPUT_TILE_SIZE_GRAD, 1);

    gradient_calc_kernel<<<dim_grid_grad, dim_block_grad>>>(gray_gauss_dev, grad_dev, width, height);

	cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess)
    {
        printf("%s\n", cudaGetErrorString(cuda_ret));
    }


    float *average_dev_1;
    float *average_dev_2;

    cudaMalloc((void**) &average_dev_1, size * sizeof(float));
    cudaMalloc((void**) &average_dev_2, size * sizeof(float));

    float * prevFrame_dev;

    cudaMalloc((void**)&prevFrame_dev, size * sizeof(float));

    if(prevFrame1 != NULL)
    {
        cudaMemcpy(prevFrame_dev, prevFrame1, size * sizeof(float), cudaMemcpyHostToDevice);
	    cuda_ret = cudaDeviceSynchronize();
	    if(cuda_ret != cudaSuccess)
        {
            printf("%s\n", cudaGetErrorString(cuda_ret));
        }
        diff_kernel<<<dim_grid_gray, dim_block_gray>>>(grad_dev, prevFrame_dev, average_dev_1, width, height);
	    cuda_ret = cudaDeviceSynchronize();
	    if(cuda_ret != cudaSuccess)
        {
            printf("%s\n", cudaGetErrorString(cuda_ret));
        }
    }
    if(prevFrame2 != NULL)
    {
        cudaMemcpy(prevFrame_dev, prevFrame2, size * sizeof(float), cudaMemcpyHostToDevice);
	    cuda_ret = cudaDeviceSynchronize();
	    if(cuda_ret != cudaSuccess)
        {
            printf("%s\n", cudaGetErrorString(cuda_ret));
        }
        diff_kernel<<<dim_grid_gray, dim_block_gray>>>(average_dev_1, prevFrame_dev, average_dev_2, width, height);
	    cuda_ret = cudaDeviceSynchronize();
	    if(cuda_ret != cudaSuccess)
        {
            printf("%s\n", cudaGetErrorString(cuda_ret));
        }
    }
    if(prevFrame3 != NULL)
    {
        cudaMemcpy(prevFrame_dev, prevFrame3, size * sizeof(float), cudaMemcpyHostToDevice);
	    cuda_ret = cudaDeviceSynchronize();
	    if(cuda_ret != cudaSuccess)
        {
            printf("%s\n", cudaGetErrorString(cuda_ret));
        }
        diff_kernel<<<dim_grid_gray, dim_block_gray>>>(average_dev_2, prevFrame3, average_dev_1, width, height);
	    cuda_ret = cudaDeviceSynchronize();
	    if(cuda_ret != cudaSuccess)
        {
            printf("%s\n", cudaGetErrorString(cuda_ret));
        }
    }

    cudaMemcpy(out, (prevFrame3 == NULL && prevFrame2 != NULL) ? average_dev_2 : (prevFrame1 == NULL) ? grad_dev : average_dev_1, size * sizeof(float), cudaMemcpyDeviceToHost);
	cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess)
    {
        printf("%s\n", cudaGetErrorString(cuda_ret));
    }

    cudaFree(grad_dev);
    cudaFree(average_dev_1);
    cudaFree(average_dev_2);
    cudaFree(prevFrame_dev);
}
