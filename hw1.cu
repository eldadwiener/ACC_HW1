/* compile with: nvcc -O3 hw1.cu -o hw1 */

#include <stdio.h>
#include <sys/time.h>

///////////////////////////////////////////////// DO NOT CHANGE ///////////////////////////////////////
#define IMG_HEIGHT 256
#define IMG_WIDTH 256
#define N_IMAGES 1

typedef unsigned char uchar;

#define CUDA_CHECK(f) do {                                                                  \
    cudaError_t e = f;                                                                      \
    if (e != cudaSuccess) {                                                                 \
        printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
        exit(1);                                                                            \
    }                                                                                       \
} while (0)

#define SQR(a) ((a) * (a))

void process_image(uchar *img_in, uchar *img_out) {
    int histogram[256] = { 0 };
    for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++) {
        histogram[img_in[i]]++;
    }

    int cdf[256] = { 0 };
    int hist_sum = 0;
    for (int i = 0; i < 256; i++) {
        hist_sum += histogram[i];
        cdf[i] = hist_sum;
    }

    int cdf_min = 0;
    for (int i = 0; i < 256; i++) {
        if (cdf[i] != 0) {
            cdf_min = cdf[i];
            break;
        }
    }

    uchar map[256] = { 0 };
    for (int i = 0; i < 256; i++) {
        int map_value = (float)(cdf[i] - cdf_min) / (IMG_WIDTH * IMG_HEIGHT - cdf_min) * 255;
        map[i] = (uchar)map_value;
    }

    for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++) {
        img_out[i] = map[img_in[i]];
    }
}

double static inline get_time_msec(void) {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec * 1e+3 + t.tv_usec * 1e-3;
}

long long int distance_sqr_between_image_arrays(uchar *img_arr1, uchar *img_arr2) {
    long long int distance_sqr = 0;
    for (int i = 0; i < N_IMAGES * IMG_WIDTH * IMG_HEIGHT; i++) {
        distance_sqr += SQR(img_arr1[i] - img_arr2[i]);
    }
    return distance_sqr;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ int arr_min(int arr[], int arr_size) {
    // assume arr_size threads call this function for arr[]
    __shared__ int SharedMin = 0;
    // TODO: DID WE UNDERSTAND CORRECTLY?!!?!? 
    int tid = threadIdx.x;
    if((arr[tid] > 0) && ((tid == 0) || (arr[tid-1] == 0))) // cdf is a rising function, so only the first non zero will have zero before it.
        {SharedMin = arr[tid];
        printf("tid - %d,SharedMin - %c",tid,SharedMin);}
    __syncthreads();
    return SharedMin; //TODO
}

__device__ void prefix_sum(int arr[], int arr_size) {
    int tid = threadIdx.x;
    int increment;
    for ( int stride = 1 ; stride <= arr_size-1 ; stride*=2 )
    {
        increment = 0;
        if(tid>=stride)
            increment = arr[tid - stride];
        __syncthreads();
        arr[tid] += increment;
        __syncthreads();
    }
    return; //TODO
}

__global__ void process_image_kernel(uchar *in, uchar *out) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    __shared__ int hist[256] = { 0 };
    for(int row = 0; row < IMG_HEIGHT; ++row)
    {
        int val = in[row][tid];
        atomicAdd(&hist[val],1);
    }
    __syncthreads();
    // hist is ready, build cdf
    prefix_sum(hist,256);
    __syncthreads();
    // hist arr now contains cdf
    int cdfMinId = arr_min(hist,256);
    // build map array
    __shared__ uchar map[256];
    map[tid] = 255 * ((hist[tid]-cdfMinId)/(IMG_HEIGHT*IMG_WIDTH - cdfMinId));
    __syncthreads();
    // create new image
    for(int row = 0; row < IMG_HEIGHT; ++row)
    {
        out[row][tid] = map[in[row][tid]];
    }
    return ; //TODO
}

int main() {
///////////////////////////////////////////////// DO NOT CHANGE ///////////////////////////////////////
    uchar *images_in;
    uchar *images_out_cpu; //output of CPU computation. In CPU memory.
    uchar *images_out_gpu_serial; //output of GPU task serial computation. In CPU memory.
    uchar *images_out_gpu_bulk; //output of GPU bulk computation. In CPU memory.
    CUDA_CHECK( cudaHostAlloc(&images_in, N_IMAGES * IMG_HEIGHT * IMG_WIDTH, 0) );
    CUDA_CHECK( cudaHostAlloc(&images_out_cpu, N_IMAGES * IMG_HEIGHT * IMG_WIDTH, 0) );
    CUDA_CHECK( cudaHostAlloc(&images_out_gpu_serial, N_IMAGES * IMG_HEIGHT * IMG_WIDTH, 0) );
    CUDA_CHECK( cudaHostAlloc(&images_out_gpu_bulk, N_IMAGES * IMG_HEIGHT * IMG_WIDTH, 0) );

    /* instead of loading real images, we'll load the arrays with random data */
    srand(0);
    for (long long int i = 0; i < N_IMAGES * IMG_WIDTH * IMG_HEIGHT; i++) {
        images_in[i] = rand() % 256;
    }

    double t_start, t_finish;

    // CPU computation. For reference. Do not change
    printf("\n=== CPU ===\n");
    t_start = get_time_msec();
    for (int i = 0; i < N_IMAGES; i++) {
        uchar *img_in = &images_in[i * IMG_WIDTH * IMG_HEIGHT];
        uchar *img_out = &images_out_cpu[i * IMG_WIDTH * IMG_HEIGHT];
        process_image(img_in, img_out);
    }
    t_finish = get_time_msec();
    printf("total time %f [msec]\n", t_finish - t_start);

    long long int distance_sqr;
///////////////////////////////////////////////////////////////////////////////////////////////////////////

    // GPU task serial computation
    printf("\n=== GPU Task Serial ===\n"); //Do not change
    //TODO: allocate GPU memory for a single input image and a single output image
    uchar* img_in, img_out;
    cudaMalloc((void**)&img_in,IMG_HEIGHT*IMG_WIDTH);
    cudaMalloc((void**)&img_out,IMG_HEIGHT*IMG_WIDTH);
    t_start = get_time_msec(); //Do not change
    //TODO: in a for loop:
    //   1. copy the relevant image from images_in to the GPU memory you allocated
    //   2. invoke GPU kernel on this image
    //   3. copy output from GPU memory to relevant location in images_out_gpu_serial
    for(int i = 0; i < N_IMAGES; ++i)
    {
        cudaMemcpy(img_in,&images_in[i * IMG_WIDTH * IMG_HEIGHT], IMG_HEIGHT * IMG_WIDTH,cudaMemcpyHostToDevice);
        dim3 threads(256),blocks(1);
        process_image_kernel<<<blocks,threads>>>(img_in,img_out);
        cudaDeviceSynchronize();
        // Check for errors
        cudaError_t error = cudaGetLastError();
        if(error != cudaSuccess){
            fprintf(stderr,"Kernel execution failed:%s\n",cudaGetErrorString(error));
            return 1;
        }
        // no error, copy image to cpu memory
        cudaMemcpy(img_out, &images_out[i * IMG_WIDTH * IMG_HEIGHT], IMG_HEIGHT * IMG_WIDTH,cudaMemcpyDeviceToHost);
    }
    t_finish = get_time_msec(); //Do not change
    distance_sqr = distance_sqr_between_image_arrays(images_out_cpu, images_out_gpu_serial); // Do not change
    printf("total time %f [msec]  distance from baseline %lld (should be zero)\n", t_finish - t_start, distance_sqr); //Do not change

    // GPU bulk
    printf("\n=== GPU Bulk ===\n"); //Do not change
    //TODO: allocate GPU memory for a all input images and all output images
    t_start = get_time_msec(); //Do not change
    //TODO: copy all input images from images_in to the GPU memory you allocated
    //TODO: invoke a kernel with N_IMAGES threadblocks, each working on a different image
    //TODO: copy output images from GPU memory to images_out_gpu_bulk
    t_finish = get_time_msec(); //Do not change
    distance_sqr = distance_sqr_between_image_arrays(images_out_cpu, images_out_gpu_bulk); // Do not change
    printf("total time %f [msec]  distance from baseline %lld (should be zero)\n", t_finish - t_start, distance_sqr); //Do not chhange

    return 0;
}
