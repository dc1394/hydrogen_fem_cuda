/*! \file sygvd_cuda.h
    \brief GPUで一般化固有値問題を解く関数の宣言

    Copyright © 2021 @dc1394 All Rights Reserved.
    This software is released under the BSD 2-Clause License.    
*/
#ifndef _SYGVD_CUDA_H_
#define _SYGVD_CUDA_H_

#include <cstdint>  // for std::int32_t
#include <utility>  // for std::pair
#include <vector>   // for std::vector

namespace cuda {
    //! A function.
    /*!
        一般化固有値問題AC=λBCを解く関数
        \param m 行の大きさ（＝列の大きさ）
        \param A 左辺の行列A
        \param B 右辺の行列B
        \return 固有値と固有ベクトルのstd::pair
    */
    [[nodiscard]]
    std::pair<std::vector<float>, std::vector<float> > sygvd_cuda(std::int32_t m, float const * A, float const * B);
}

#endif // _SYGVD_CUDA_H_
