#ifndef __KERNELS
#define __KERNELS
#define FILTER_SIZE 5
#define OUTPUT_TILE_SIZE 12
#define INPUT_TILE_SIZE (OUTPUT_TILE_SIZE + FILTER_SIZE - 1)

#define FILTER_SIZE_GRAD 3
#define INPUT_TILE_SIZE_GRAD (OUTPUT_TILE_SIZE + FILTER_SIZE_GRAD - 1)

#define OVERHANG ((INPUT_TILE_SIZE - OUTPUT_TILE_SIZE) / 2)
#define OVERHANG_GRAD ((INPUT_TILE_SIZE_GRAD - OUTPUT_TILE_SIZE) / 2)

__constant__ float gauss_filter[FILTER_SIZE][FILTER_SIZE] =
{{2/115, 4/115, 5/115, 4/115, 2/115},
 {4/115, 9/115, 12/115, 9/115, 4/115},
 {5/115, 12/115, 15/115, 12/115, 5/115},
 {4/115, 9/115, 12/115, 9/115, 4/115},
 {2/115, 4/115, 5/115, 4/115, 2/115}};

__constant__ float Gx_filter[FILTER_SIZE_GRAD][FILTER_SIZE_GRAD] = 
{{-1, 0, 1},
 {-2, 0, 2},
 {-1, 0, 1}};

__constant__ float Gy_filter[FILTER_SIZE_GRAD][FILTER_SIZE_GRAD] =
{{1, 2, 1},
 {0, 0, 0},
 {-1, -2. -1}};

__global__ void grayscale_kernel(float * r, float * g, float * b, float * out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < width && y < height)
    {
        out[y * width + x] = (r[y * width + x] + g[y * width + x] + b[y * width + x]) / 3;
    }
}

__global__ void gaussian_filter_kernel(float * in, float * out, int width, int height) {
    
    //initialize a shared bit of memory for collaborative loading and all that shit
    __shared__ float collab[INPUT_TILE_SIZE][INPUT_TILE_SIZE];

    //indices into output image
    int output_x = blockIdx.x * OUTPUT_TILE_SIZE + threadIdx.x - OVERHANG;
    int output_y = blockIdx.y * OUTPUT_TILE_SIZE + threadIdx.y - OVERHANG;

    collab[threadIdx.y][threadIdx.x] = (output_x >= 0 && output_x < width
                                        && output_y >= 0 && output_y < height) ? 
                                        in[output_y * width + output_x] : 0;
    
    __syncthreads();

    if((int) threadIdx.x - OVERHANG >= 0 &&
       threadIdx.x - OVERHANG < OUTPUT_TILE_SIZE &&
       (int) threadIdx.y - OVERHANG >= 0 &&
       threadIdx.y - OVERHANG < OUTPUT_TILE_SIZE &&
       output_x < width && output_y < height)
    {
        float accum = 0.0f;
        for(int i = 0; i < FILTER_SIZE; i++)
        {
            for(int j = 0; j < FILTER_SIZE; j++)
            {
                accum += collab[threadIdx.y + i - OVERHANG][threadIdx.x + j - OVERHANG] * gauss_filter[i][j];
            }
        }

        out[output_y * width + output_x] = accum;
    }
}

__global__ void gradient_calc_kernel(float * in, float * Gx, float * Gy, float * G, int width, int height) {
    //initialize an area of shared memeory
    __shared__ float collab[INPUT_TILE_SIZE_GRAD][INPUT_TILE_SIZE_GRAD];

    int output_x = blockIdx.x * OUTPUT_TILE_SIZE + threadIdx.x - OVERHANG_GRAD;
    int output_y = blockIdx.y * OUTPUT_TILE_SIZE + threadIdx.y - OVERHANG_GRAD;

    collab[threadIdx.y][threadIdx.x] = (output_x >= 0 && output_x < width
                                        && output_y >= 0 && output_y < height) ?
                                        in[output_y * width + output_x] : 0;

    __syncthreads();

    if((int)threadIdx.x - OVERHANG_GRAD >= 0 &&
        threadIdx.x - OVERHANG_GRAD < OUTPUT_TILE_SIZE &&
        (int)threadIdx.y - OVERHANG_GRAD >= 0 &&
        threadIdx.y - OVERHANG < OUTPUT_TILE_SIZE &&
        output_x < width && output_y < height)
    {
        float accumX = 0.0f;
        float accumY = 0.0f;

        for(int i = 0; i < FILTER_SIZE_GRAD; i++)
        {
            for(int j = 0; j < FILTER_SIZE_GRAD; j++)
            {
                accumX += collab[threadIdx.x + i - OVERHANG_GRAD][threadIdx.x + j - OVERHANG_GRAD] * Gx_filter[i][j];
                accumY += collab[threadIdx.x + i - OVERHANG_GRAD][threadIdx.x + j - OVERHANG_GRAD] * Gy_filter[i][j];
            }
        }
        
        Gx[output_y * width + output_x] = accumX;
        Gy[output_y * width + output_x] = accumY;
        G[output_y * width + output_x] = abs(accumX) + abs(accumY);
    }

}

__global__ void theta_calc_kernel(float * gradX, float * gradY, int * out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < width && y < height)
    {
        float val = atan2(gradY[y * width + x], gradX[y * width + x]);
        val /= 3.1415926535;//PI
        //now we've constrained ourselves to a range of [-1, 1]
        if(val >= -.125 && val < .125)
            out[y * width + x] = 0;
        else if(val >= .125 && val < .375)
            out[y * width + x] = 1;
        else if(val >= .375 && val < .625)
            out[y * width + x] = 2;
        else if(val >= .625 && val < .875)
            out[y * width + x] = 3;
        else if(val >= .875 || val < -.875)
            out[y * width + x] = 4;
        else if(val >= -.875 && val < -.625)
            out[y * width + x] = 5;
        else if(val >= -.625 && val < -.375)
            out[y * width + x] = 6;
        else
            out[y * width + x] = 7;
    }
}
#endif //__KERNELS
