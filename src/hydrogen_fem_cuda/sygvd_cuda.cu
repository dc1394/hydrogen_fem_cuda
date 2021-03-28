/*! \file sygvd_cuda.h
    \brief GPUで一般化固有値問題を解く関数の実装

    Copyright © 2021 @dc1394 All Rights Reserved.
    This software is released under the BSD 2-Clause License.    
*/

#include "sygvd_cuda.h"
#include <cassert>          // for assert 
#include <cusolverDn.h>

namespace cuda {
    std::pair<std::vector<float>, std::vector<float> > sygvd_cuda(std::int32_t m, float const * A, float const * B)
    {
        cusolverDnHandle_t cusolverH = nullptr;
        cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
        cudaError_t cudaStat1 = cudaSuccess;
        cudaError_t cudaStat2 = cudaSuccess;
        cudaError_t cudaStat3 = cudaSuccess;
        cudaError_t cudaStat4 = cudaSuccess;
        const int lda = m;

        std::vector<float> V(lda*m); // eigenvectors
        std::vector<float> W(m); // eigenvalues

        float *d_A = nullptr;
        float *d_B = nullptr;
        float *d_W = nullptr;
        int *devInfo = nullptr;
        float *d_work = nullptr;
        int  lwork = 0;
        int info_gpu = 0;
    
        // step 1: create cusolver/cublas handle
        cusolver_status = cusolverDnCreate(&cusolverH);
        assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

        // step 2: copy A and B to device
        cudaStat1 = cudaMalloc ((void**)&d_A, sizeof(float) * lda * m);
        cudaStat2 = cudaMalloc ((void**)&d_B, sizeof(float) * lda * m);
        cudaStat3 = cudaMalloc ((void**)&d_W, sizeof(float) * m);
        cudaStat4 = cudaMalloc ((void**)&devInfo, sizeof(int));
        assert(cudaSuccess == cudaStat1);
        assert(cudaSuccess == cudaStat2);
        assert(cudaSuccess == cudaStat3);
        assert(cudaSuccess == cudaStat4);

        cudaStat1 = cudaMemcpy(d_A, A, sizeof(float) * lda * m, cudaMemcpyHostToDevice);
        cudaStat2 = cudaMemcpy(d_B, B, sizeof(float) * lda * m, cudaMemcpyHostToDevice);
        assert(cudaSuccess == cudaStat1);
        assert(cudaSuccess == cudaStat2);

        // step 3: query working space of sygvd
        cusolverEigType_t itype = CUSOLVER_EIG_TYPE_1; // A*x = (lambda)*B*x
        cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
        cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
        cusolver_status = cusolverDnSsygvd_bufferSize(
            cusolverH,
            itype,
            jobz,
            uplo,
            m,
            d_A,
            lda,
            d_B,
            lda,
            d_W,
            &lwork);
        assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
        cudaStat1 = cudaMalloc((void**)&d_work, sizeof(float)*lwork);
        assert(cudaSuccess == cudaStat1);

        // step 4: compute spectrum of (A,B)
        cusolver_status = cusolverDnSsygvd(
            cusolverH,
            itype,
            jobz,
            uplo,
            m,
            d_A,
            lda,
            d_B,
            lda,
            d_W,
            d_work,
            lwork,
            devInfo);
        cudaStat1 = cudaDeviceSynchronize();
        assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
        assert(cudaSuccess == cudaStat1);
    
        cudaStat1 = cudaMemcpy(W.data(), d_W, sizeof(float)*m, cudaMemcpyDeviceToHost);
        cudaStat2 = cudaMemcpy(V.data(), d_A, sizeof(float)*lda*m, cudaMemcpyDeviceToHost);
        cudaStat3 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
        assert(cudaSuccess == cudaStat1);
        assert(cudaSuccess == cudaStat2);
        assert(cudaSuccess == cudaStat3);

        assert(0 == info_gpu);

        // free resources
        if (d_A    ) cudaFree(d_A);
        if (d_B    ) cudaFree(d_B);
        if (d_W    ) cudaFree(d_W);
        if (devInfo) cudaFree(devInfo);
        if (d_work ) cudaFree(d_work);

        if (cusolverH) cusolverDnDestroy(cusolverH);

        cudaDeviceReset();

        return std::make_pair(std::move(W), std::move(V));
    }
}