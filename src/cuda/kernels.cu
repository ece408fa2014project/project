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
{{(float)2/159, (float)4/159,  (float)5/159,  (float)4/159,  (float)2/159},
 {(float)4/159, (float)9/159,  (float)12/159, (float)9/159,  (float)4/159},
 {(float)5/159, (float)12/159, (float)15/159, (float)12/159, (float)5/159},
 {(float)4/159, (float)9/159,  (float)12/159, (float)9/159,  (float)4/159},
 {(float)2/159, (float)4/159,  (float)5/159,  (float)4/159,  (float)2/159}};

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

    __shared__ float collab[INPUT_TILE_SIZE][INPUT_TILE_SIZE];

    float *N_data = in;
    float *P_data = out;

    //these are the indices that reference the output array
    int output_x = blockIdx.x * OUTPUT_TILE_SIZE + threadIdx.x - OVERHANG;
    int output_y = blockIdx.y * OUTPUT_TILE_SIZE + threadIdx.y - OVERHANG;

    collab[threadIdx.y][threadIdx.x] = (output_x >= 0 && output_x < width
                                     && output_y >= 0 && output_y < height) ? N_data[output_y * width + output_x] : 0;

    __syncthreads();

    if((int)threadIdx.x - OVERHANG >= 0 &&
       (int)threadIdx.x - OVERHANG < OUTPUT_TILE_SIZE &&
       (int)threadIdx.y - OVERHANG >= 0 &&
       (int)threadIdx.y - OVERHANG < OUTPUT_TILE_SIZE &&
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
        P_data[output_y * width + output_x] = accum;
    }
}

__global__ void gradient_calc_kernel(float * in, float * G, int width, int height) {

    __shared__ float collab[INPUT_TILE_SIZE_GRAD][INPUT_TILE_SIZE_GRAD];

    float *N_data = in;
    float *P_data = G;

    //these are the indices that reference the output array
    int output_x = blockIdx.x * OUTPUT_TILE_SIZE + threadIdx.x - OVERHANG_GRAD;
    int output_y = blockIdx.y * OUTPUT_TILE_SIZE + threadIdx.y - OVERHANG_GRAD;

    collab[threadIdx.y][threadIdx.x] = (output_x >= 0 && output_x < width
                                     && output_y >= 0 && output_y < height) ? N_data[output_y * width + output_x] : 0;

    __syncthreads();

    if((int)threadIdx.x - OVERHANG_GRAD >= 0 &&
       (int)threadIdx.x - OVERHANG_GRAD < OUTPUT_TILE_SIZE &&
       (int)threadIdx.y - OVERHANG_GRAD >= 0 &&
       (int)threadIdx.y - OVERHANG_GRAD < OUTPUT_TILE_SIZE &&
       output_x < width && output_y < height)
    {
        float accumX = 0.0f;
        float accumY = 0.0f;
        for(int i = 0; i < FILTER_SIZE_GRAD; i++)
        {
            for(int j = 0; j < FILTER_SIZE_GRAD; j++)
            {
                accumX += collab[threadIdx.y + i - OVERHANG_GRAD][threadIdx.x + j - OVERHANG_GRAD] * Gx_filter[i][j];
                accumY += collab[threadIdx.y + i - OVERHANG_GRAD][threadIdx.x + j - OVERHANG_GRAD] * Gy_filter[i][j];
            }
        }
        P_data[output_y * width + output_x] = abs(accumX) + abs(accumY);
        if(P_data[output_y * width + output_x] >= 255)
            P_data[output_y * width + output_x] = 255;
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

__global__ void diff_kernel(float *orig, float *comp, float *out, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < width && y < height)
    {
        out[y * width + x] = abs(orig[y * width + x] - comp[y * width + x] / 3);
    }
}

__global__ void skin_detection(float *  R, float *  G, float *  B, float *  retR, float *  retG,  int width, int height) {
    //INSERT KERNEL CODE HERE
    float r, g, Red, Green, Blue, F1, F2, w, theta, H, I;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = blockIdx.y * blockDim.y + ty;
    int col_o = blockIdx.x * blockDim.x + tx;

    if(row_o < height && col_o < width)
    {
        /*get the RGB value*/
        Red = R[row_o * width + col_o];
        Green = G[row_o * width + col_o];
        Blue = B[row_o * width + col_o];
        /*get the intensity*/
        I = (Red + Green + Blue)/3;
        /*normalized red and green*/
        r = Red/(Red + Green + Blue);
        g = Green/(Red + Green + Blue);
        /*function 1 and 2 and w in the doc*/
        F1 = -1.376 * r * r + 1.0743 * r + 0.2;
        F2 = -0.776 * r * r + 0.5601 * r + 0.18;
        w = (r - 0.33) * (r-0.33) + (g - 0.33) * (g - 0.33);
        theta = acos(0.5 * (Red * 2 - Green - Blue) / sqrt((Red - Green) * (Red - Green) + (Red - Blue) * (Green - Blue)));
        if (Blue <= Green)
            H = theta;
        else
            H = 3.1415926535 * 2 - theta;
        /*face detected*/
        if(g < F1 && g > F2 && w > 0.001 && ((H > (3.1415926535*4/3)) || (H < (3.1415926535 / 9))))
        /*set R to be 255*/
            retR[row_o * width + col_o] = 255;
        else
            retR[row_o * width + col_o] = 0;
        /*hair detected*/
        if( I < 80 && (Blue - Green < 15 || Blue - Red < 15))
            //if ((H <= (2*3.1415926535/9)) && H > (3.1415926535 /9))
            retG[row_o * width + col_o] = 255;
        else
            retG[row_o * width + col_o] = 0;

    }
}

__global__ void clean_up(float *  R, float *  G, float * retR, float * retG, int width, int height) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = blockIdx.y * blockDim.y + ty;
    int col_o = blockIdx.x * blockDim.x + tx;
    int i,j;
    int counter1 = 0;
    int counter2 = 0;
    if(R[row_o * width + col_o] == 255)
    {
        for(i = -2; i < 3; i++)
        {
            for(j = -2; j < 3; j++)
            {
                if(((row_o + j) * width + col_o+i) < 0)
                {
                    if(R[0] == 255)
                        counter1++;
                }
                else if(R[(row_o + j) * width + col_o+i] == 255)
                    counter1++;
            }
        }
    }
    if(G[row_o * width + col_o] == 255)
    {
        for(i = -3; i < 4; i++)
        {
            for(j = -3; j < 4; j++)
            {
                if(((row_o + j) * width + col_o+i) < 0)
                {
                    if(G[0] == 255)
                        counter2++;
                }
                else if(G[(row_o + j) * width + col_o+i] == 255)
                    counter2++;
            }
        }
    }
    if(counter1 >20)
        retR[row_o * width + col_o] = 255;
    else
        retR[row_o * width + col_o] = 0;
    if(counter2 >20)
        retG[row_o * width + col_o] = 255;
    else
        retG[row_o * width + col_o] = 0;

}
__global__ void quantization(float *  R, float *  G, float * retR, float * retG, int width, int height) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = blockIdx.y * blockDim.y + ty;
    int col_o = blockIdx.x * blockDim.x + tx;
    int i,j;
    if(R[row_o * width + col_o] == 255)
    {
        for(i = -2; i < 3; i++)
        {
            for(j = -2; j < 3; j++)
            {
                if(((row_o + j) * width + col_o+i) <= 0)
                {
                    retR[0] = 255;
                }
                else
                    retR[(row_o + j) * width + col_o+i] = 255;
            }
        }
    }
    else if(G[row_o * width + col_o] == 255)
    {
        for(i = -2; i < 3; i++)
        {
            for(j = -2; j < 3; j++)
            {
                if(((row_o + j) * width + col_o+i) <= 0)
                    retG[0] = 0;
                else
                    retG[(row_o + j) * width + col_o+i] = 255;

            }
        }
    }
}

void face_detection(float * R, float * G, float * r_out, float * g_out,float * b_out,int width, int height) {

    int i, j;
    int min_x_r = width-1;
    int max_x_r = 0;
    int min_y_r = height-1;
    int max_y_r = 0;
    int min_x_g = width-1;
    int max_x_g = 0;
    int min_y_g = height-1;
    int max_y_g = 0;
    for(i = 0; i < width; i++)
    {
        for(j = 0; j < height; j++)
        {
            /*skin detection*/
            if(R[j* width + i] == 255)
            {
                //printf("%d\n", j);
                if(j < min_y_r)
                    min_y_r = j;
                else if(j > max_y_r)
                    max_y_r = j;
                if(i < min_x_r)
                    min_x_r = i;
                else if(i > max_x_r)
                    max_x_r = i;
            }
        }
    }
    for(i = 0; i < width; i++)
    {
        for(j = 0; j < height; j++)
        {
            /*skin detection*/
            if(G[j* width + i] == 255)
            {
                //printf("%d\n", j);
                if(j < min_y_g)
                    min_y_g = j;
                else if(j > max_y_g)
                    max_y_g = j;
                if(i < min_x_g)
                    min_x_g = i;
                else if(i > max_x_g)
                    max_x_g = i;
            }
        }
    }
    /*draw the box*/
    /*draw the box*/
    //if(min_y_r > min_y_g && min_x_r > min_x_g && max_x_r < max_x_g)
    //{
    //printf("%d\n", min_x_r);
    // printf("%d\n", max_x_r);
    // printf("%d\n", min_y_r);
    // printf("%d\n", max_x_r);
    if(min_x_r >= min_x_g && min_y_r >= min_y_g && max_x_r <= max_x_g)
    {
        for(i = min_x_r; i < max_x_r; i++)
        {
            r_out[max_y_r * width + i] = 255;
            g_out[max_y_r * width + i] = 0;
            b_out[max_y_r * width + i] = 0;
            r_out[min_y_r * width + i] = 255;
            g_out[min_y_r * width + i] = 0;
            b_out[min_y_r * width + i] = 0;
        }
        for(i = min_y_r; i < max_y_r; i++)
        {
            r_out[i * width + min_x_r] = 255;
            g_out[i * width + min_x_r] = 0;
            b_out[i * width + min_x_r] = 0;
            r_out[i * width + max_x_r] = 255;
            g_out[i * width + max_x_r] = 0;
            b_out[i * width + max_x_r] = 0;
        }
    }
    //image->writeToFile("poop.png");

    for(i = min_x_r; i < max_x_r; i++)
    {
        R[max_y_r * width + i] = 255;
        R[min_y_r * width + i] = 255;
    }
    for(i = min_y_r; i < max_y_r; i++)
    {
        R[i * width + min_x_r] = 255;
        R[i * width + max_x_r] = 255;
    }


}

#endif //__KERNELS
