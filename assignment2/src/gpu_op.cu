#include "./c_runtime_api.h"
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

int divupround(int a, int b) {
  if(a % b == 0) return a / b;
  return a / b + 1;
}

int64_t totallength(DLArrayHandle array) {
  int64_t length = 1;
  for(int i = 0; i < array->ndim; i++) {
    length *= array->shape[i];
  }
  return length;
}

/* TODO: Your code here */
/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void matrix_softmax_cross_entropy_kernel(int nrow, int ncol,
                                                    const float *input_a,
                                                    const float *input_b,
                                                    float *output) {
  // Dynamic shared memory, size provided at kernel launch.
  extern __shared__ float loss_per_row[];
  // Two dimensional thread blocks.
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
          threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input_a += y * ncol;
  input_b += y * ncol;
  float maxval = *input_a;
  // Find max for a row.
  for (int x = 1; x < ncol; ++x) {
    maxval = max(maxval, input_a[x]);
  }
  // Deduct by max for a row, and raise to exp.
  float sum = 0;
  for (int x = 0; x < ncol; ++x) {
    sum += exp(input_a[x] - maxval);
  }
  // Compute per-row loss.
  float loss = 0;
  for (int x = 0; x < ncol; ++x) {
    loss -= input_b[x] * log(exp(input_a[x] - maxval) / sum);
  }
  loss_per_row[y] = loss;
  __syncthreads();
  // Compute reduce_mean across rows.
  float mean_loss = 0;
  // Use a single thread to reduce mean across rows.
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    for (int i = 0; i < nrow; ++i) {
      mean_loss += loss_per_row[i];
    }
    mean_loss /= nrow;
    output[0] = mean_loss;
  }
}

__global__ void matrix_softmax_kernel(int nrow, 
                                      int ncol, 
                                      const float *input_data, 
                                      float *output_data){
  // Two dimensional thread blocks.
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
          threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input_data += y * ncol;
  output_data += y * ncol;
  float maxval = *input_data;
  // Find max for a row.
  for (int x = 1; x < ncol; ++x) {
    maxval = max(maxval, input_data[x]);
  }
  // Deduct by max for a row, and raise to exp.
  float sum = 0;
  for (int x = 0; x < ncol; ++x) {
    sum += exp(input_data[x] - maxval);
  }

  for (int x = 0; x < ncol; ++x) {
    output_data[x] = exp(input_data[x] - maxval) / sum;
  }
}

__global__ void relu_kernel(int64_t length, 
                            const float *input_data,
                            float *output_data){
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y > length){
      return;
    }
    output_data[y] = max(0.0f, input_data[y]);
}

__global__ void relu_gradient_kernel(int64_t length,
                                     const float *input_data,
                                     const float *in_grad_data,
                                     float *output_data){
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y > length){
    return;
  }
  output_data[y] = input_data[y]>0.0f? in_grad_data[y]:0.0f;
}

__global__ void array_set_kernel(int64_t length,
                                 float *array_data,
                                 float value){
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y > length){
    return;
  }
  array_data[y] = value;
}

__global__ void broadcast_kernel(int64_t length, 
                                 const float *input_data,
                                 float *output_data){
  output_data += length * blockIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if(x < length) {
    output_data[x] = input_data[x];
  }
}

__global__ void reduce_sum_axis_zero_kernel(int64_t output_length, int reduce_size, const float* input, float *output) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  if(x >= output_length) return;
  float value = 0;
  for(int i = threadIdx.y; i < reduce_size; i+= blockDim.y) {
    value += input[i * output_length + x];
  }
  atomicAdd(output + x, value);
}

__global__ void matrix_add_kernel(int64_t length,
                                  const float *matA_data, 
                                  const float *matB_data, 
                                  float *output_data){
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y > length){
    return;
  }
  output_data[y] = matA_data[y] + matB_data[y]; 
}

__global__ void matrix_add_by_const_kernel(int64_t length,
                                           const float *input_data,
                                           float val,
                                           float *output_data){
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y > length){
    return;
  }
  output_data[y] = input_data[y] + val; 
}

__global__ void matrix_mul_kernel(int64_t length,
                                  const float *matA_data, 
                                  const float *matB_data, 
                                  float *output_data){
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y > length){
    return;
  }
  output_data[y] = matA_data[y] * matB_data[y]; 
}

__global__ void matrix_mul_by_const_kernel(int64_t length,
                                           const float *input_data,
                                           float val,
                                           float *output_data){
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y > length){
    return;
  }
  output_data[y] = input_data[y] * val; 
}

cublasHandle_t cublas_handle = NULL;

int DLGpuArraySet(DLArrayHandle arr, float value) { /* TODO: Your code here */
  int64_t length = totallength(arr);
  float *array_data = (float *)arr->data;
  dim3 DimGrid((length-1)/1024+1, 1, 1);
  dim3 DimBlock(1024, 1, 1);
  array_set_kernel<<<DimGrid, DimBlock>>>(length, array_data, value);  
  return 0;
}

int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  int64_t length = totallength(input);
  const float *input_data = (float *)input->data;
  float *output_data = (float *)output->data;
  dim3 threads;
  if (length < 1024){
    threads.x = (int)length;
  }else{
    threads.x = 1024;
    threads.y = (int)((length + 1023)/1024);
  }
  broadcast_kernel<<<dim3(divupround(length, 1024), output->shape[0]), 1024>>>(length, input_data, output_data);
  return 0;
}

int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  DLGpuArraySet(output, 0);
  int output_length = totallength(output);
  reduce_sum_axis_zero_kernel<<<divupround(output_length, 64), dim3(min(64, output_length), 16)>>>(output_length, input->shape[0], (float*)input->data, (float*)output->data);
  return 0;
}

int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                              const DLArrayHandle matB, DLArrayHandle output) {
  /* TODO: Your code here */
  int64_t length = totallength(output);
  const float *matA_data = (const float *)matA->data;
  const float *matB_data = (const float *)matB->data;
  float *output_data = (float *)output->data;
  dim3 DimGrid((length-1)/1024+1, 1, 1);
  dim3 DimBlock(1024, 1, 1);
  matrix_add_kernel<<<DimGrid, DimBlock>>>(length, matA_data, matB_data, output_data);
  return 0;
}

int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                     DLArrayHandle output) {
  /* TODO: Your code here */
  int64_t length = totallength(output);
  const float *input_data = (const float *)input->data;
  float *output_data = (float *)output->data;
  dim3 DimGrid((length-1)/1024+1, 1, 1);
  dim3 DimBlock(1024, 1, 1);
  matrix_add_by_const_kernel<<<DimGrid, DimBlock>>>(length, input_data, val, output_data);
  return 0;
}

int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA,
                                   const DLArrayHandle matB,
                                   DLArrayHandle output) {
  /* TODO: Your code here */
  int64_t length = totallength(output);
  const float *matA_data = (const float *)matA->data;
  const float *matB_data = (const float *)matB->data;
  float *output_data = (float *)output->data;
  dim3 DimGrid((length-1)/1024+1, 1, 1);
  dim3 DimBlock(1024, 1, 1);
  matrix_mul_kernel<<<DimGrid, DimBlock>>>(length, matA_data, matB_data, output_data);
  return 0;
}

int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                               DLArrayHandle output) {
  /* TODO: Your code here */
  int64_t length = totallength(output);
  const float *input_data = (const float *)input->data;
  float *output_data = (float *)output->data;
  dim3 DimGrid((length-1)/1024+1, 1, 1);
  dim3 DimBlock(1024, 1, 1);
  matrix_mul_by_const_kernel<<<DimGrid, DimBlock>>>(length, input_data, val, output_data);
  return 0;
}

int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                        const DLArrayHandle matB, bool transposeB,
                        DLArrayHandle matC) {
  /* TODO: Your code here */
  // Hint: use cublas
  // cublas assume matrix is column major
  if(!cublas_handle) {
    cublasCreate(&cublas_handle);
  }

  float one = 1.0f;
  float zero = 0.0f;
  int m = matC->shape[1];
  int n = matC->shape[0];
  int k = transposeA ? matA->shape[0] : matA->shape[1];

  cublasSgemm(cublas_handle,
    transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
    transposeA ? CUBLAS_OP_T : CUBLAS_OP_N,
    m, n, k,
    &one,
    (const float*)matB->data, !transposeB ? m : k,
    (const float*)matA->data, !transposeA ? k : n,
    &zero,
    (float*)matC->data, m
  );
  return 0;
}

int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  int64_t length =  totallength(output);
  const float *input_data = (const float *)input->data;
  float *output_data = (float *)output->data;
  dim3 DimGrid((length-1)/1024+1, 1, 1);
  dim3 DimBlock(1024, 1, 1);
  relu_kernel<<<DimGrid, DimBlock>>>(length, input_data, output_data);
  return 0;
}

int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                      DLArrayHandle output) {
  /* TODO: Your code here */
  int64_t length =  totallength(output);
  const float *input_data = (const float *)input->data;
  const float *in_grad_data = (const float *)in_grad->data;
  float *output_data = (float *)output->data;
  dim3 DimGrid((length-1)/1024+1, 1, 1);
  dim3 DimBlock(1024, 1, 1); 
  relu_gradient_kernel<<<DimGrid, DimBlock>>>(length, input_data, in_grad_data, output_data);
  return 0;
}

int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  //assert(input->ndim == 2);
  //assert(output->ndim == 1);
  int nrow = input->shape[0];
  assert(nrow <= 1024 * 4);
  int ncol = input->shape[1];
  const float *input_data = (const float *)input->data;
  float *output_data = (float *)output->data;
  dim3 threads;
  if (nrow < 1024){
    threads.x = nrow;
  }else{
    threads.x = 1024;
    threads.y = (nrow + 1023)/1024;
  }
  matrix_softmax_kernel<<<1, threads, nrow * sizeof(float)>>>(nrow, ncol, input_data, output_data);
  return 0;
}

int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
                             const DLArrayHandle input_b,
                             DLArrayHandle output) {
  assert(input_a->ndim == 2);
  assert(input_b->ndim == 2);
  assert(output->ndim == 1);
  assert(input_a->shape[0] == input_b->shape[0] &&
         input_a->shape[1] == input_b->shape[1]);
  int nrow = input_a->shape[0];
  // Maximum x- or y-dimension of a block = 1024
  // But we need 'nrow' shared memory, and max shared memory is 48KB.
  // Conservatively allow max 16KB shared memory.
  assert(nrow <= 1024 * 4);
  int ncol = input_a->shape[1];
  const float *input_data_a = (const float *)input_a->data;
  const float *input_data_b = (const float *)input_b->data;
  float *output_data = (float *)output->data;
  dim3 threads;
  if (nrow <= 1024) {
    threads.x = nrow;
  } else {
    threads.x = 1024;
    threads.y = (nrow + 1023) / 1024;
  }
  // 1 block, each block with 'threads' number of threads with 'nrow' shared
  // memory size
  matrix_softmax_cross_entropy_kernel<<<1, threads, nrow * sizeof(float)>>>(
      nrow, ncol, input_data_a, input_data_b, output_data);
  return 0;
}
