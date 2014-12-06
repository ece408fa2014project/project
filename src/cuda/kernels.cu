#ifndef __KERNELS
#define __KERNELS
#define FILTER_SIZE 5
#define OUTPUT_TILE_SIZE 12
#define INPUT_TILE_SIZE (OUTPUT_TILE_SIZE + FILTER_SIZE - 1)


#define OVERHANG ((INPUT_TILE_SIZE - OUTPUT_TILE_SIZE) / 2)

__constant__ float M_c[FILTER_SIZE][FILTER_SIZE] =
{{2/115, 4/115, 5/115, 4/115, 2/115},
 {4/115, 9/115, 12/115, 9/115, 4/115},
 {5/115, 12/115, 15/115, 12/115, 5/115},
 {4/115, 9/115, 12/115, 9/115, 4/115},
 {2/115, 4/115, 5/115, 4/115, 2/115}};

__global__ void grayscale_kernel(float * r, float * g, float * b, float * out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < width && y < height)
    {
        out[y * width + x] = (r[y * width + x] + g[y * width + x] + b[y * width + x]) / 3;
    }
}



//__global__ void convolution(Matrix N, Matrix P) {
//
//	/************************************************************************
//     * Determine input and output indexes of each thread                    *
//     * Load a tile of the input image to shared memory                      *
//     * Apply the filter on the input image tile                             *
//     * Write the compute values to the output image at the correct indexes  *
//     * Use OUTPUT_TILE_SIZE and INPUT_TILE_SIZE defined in support.h        *
//	 ************************************************************************/
//    //The definition of the OVERHANG constant above is to help with the repositioning
//    //of the M matrix in space.
//
//    //INSERT KERNEL CODE HERE
//    __shared__ float collab[INPUT_TILE_SIZE][INPUT_TILE_SIZE];
//    
//    int width = N.width;
//    int height = N.height; 
//    
//    float *N_data = N.elements;
//    float *P_data = P.elements;
//    
//    //these are the indices that reference the output array
//    int output_x = blockIdx.x * OUTPUT_TILE_SIZE + threadIdx.x - OVERHANG;
//    int output_y = blockIdx.y * OUTPUT_TILE_SIZE + threadIdx.y - OVERHANG;
//
//    collab[threadIdx.y][threadIdx.x] = (output_x >= 0 && output_x < width
//                                     && output_y >= 0 && output_y < height) ? N_data[output_y * width + output_x] : 0;
//
//    __syncthreads();
//
//    if(threadIdx.x - OVERHANG >= 0 &&
//       threadIdx.x - OVERHANG < OUTPUT_TILE_SIZE &&
//       threadIdx.y - OVERHANG >= 0 &&
//       threadIdx.y - OVERHANG < OUTPUT_TILE_SIZE &&
//       output_x < width && output_y < height)
//    {
//        float accum = 0.0f;
//        for(int i = 0; i < FILTER_SIZE; i++)
//        {
//            for(int j = 0; j < FILTER_SIZE; j++)
//            {
//                accum += collab[threadIdx.y + i - OVERHANG][threadIdx.x + j - OVERHANG] * M_c[i][j]; 
//            }
//        }
//        P_data[output_y * width + output_x] = accum;
//    }
//}

#endif //__KERNELS
