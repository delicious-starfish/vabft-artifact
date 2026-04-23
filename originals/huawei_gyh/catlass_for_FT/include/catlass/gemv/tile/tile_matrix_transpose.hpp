#ifndef CATLASS_GEMV_TILE_TILE_MATRIX_TRANSPOSE_HPP_SELF
#define CATLASS_GEMV_TILE_TILE_MATRIX_TRANSPOSE_HPP_SELF

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemv/helper.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/gemm_type.hpp"

namespace Catlass::Gemv::Tile {

template <
    /// Tag indicating architecture
    class ArchTag,
    class AType,
    class YType,
    class BiasType = void
>
struct TileMatrixTranspose
{
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported TileMatmulAdd, can not find the specialization.");
};

template <
    class ElementA,
    class ElementY
>
struct TileMatrixTranspose<Arch::AtlasA2,
                Gemm::GemmType<ElementA, layout::RowMajor>,
                Gemm::GemmType<ElementY, layout::RowMajor>,
                void>
{
    static constexpr uint32_t H_UNIT = 16;
    static constexpr uint32_t W_UNIT = 16;
};

template <>
struct TileMatrixTranspose<Arch::AtlasA2,
                Gemm::GemmType<half, layout::RowMajor>,
                Gemm::GemmType<half, layout::RowMajor>,
                void>
{
    static constexpr uint32_t H_UNIT = 16;
    static constexpr uint32_t W_UNIT = 16;

    using ElementA = half;
    using ElementX = half;
    using ElementY = half;

    using ElementAccumulator = ElementY;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);
    static constexpr uint32_t OUT_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);

    // Methods

    CATLASS_DEVICE
    TileMatrixTranspose() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        AscendC::LocalTensor<ElementA> temp_workspace,
        LayoutDst const &layoutDst, 
        LayoutSrc const &layoutSrc
    )
    {
        uint32_t raw_m_actual = layoutSrc.shape(0);
        uint32_t raw_n_actual = layoutSrc.shape(1);
        // RoundUp(n_round, repeat_size)
        raw_m_actual = RoundUp(raw_m_actual, H_UNIT);
        raw_n_actual = RoundUp(raw_n_actual, W_UNIT);

        uint32_t transed_m_actual = raw_n_actual;
        uint32_t transed_n_actual = raw_m_actual;

        uint32_t raw_m_round = layoutDst.shape(1);
        uint32_t raw_n_round = layoutDst.shape(0);

        uint32_t transed_m_round = raw_n_round;
        uint32_t transed_n_round = raw_m_round;

        uint32_t repeat_num_m = raw_m_actual / H_UNIT;
        uint32_t repeat_num_n = raw_n_actual / W_UNIT;

        float to_temp_scalar = 0.0f;

        uint32_t temp_repeat_times = raw_m_actual;
        uint32_t temp_mask = W_UNIT;
        AscendC::UnaryRepeatParams cache_params;
        cache_params.dstBlkStride = 1;
        cache_params.srcBlkStride = 1;
        cache_params.dstRepStride = RoundUp(W_UNIT, ELE_NUM_PER_C0) / ELE_NUM_PER_C0;
        cache_params.srcRepStride = RoundUp(raw_n_round, ELE_NUM_PER_C0) / ELE_NUM_PER_C0;

        uint32_t final_mask= H_UNIT;
        uint32_t final_repeat_times = repeat_num_m;

        AscendC::UnaryRepeatParams final_transed_params;
        final_transed_params.dstBlkStride = 1;
        final_transed_params.srcBlkStride = 1;
        final_transed_params.dstRepStride = RoundUp(H_UNIT, OUT_ELE_NUM_PER_C0) / OUT_ELE_NUM_PER_C0;
        final_transed_params.srcRepStride = RoundUp(H_UNIT * W_UNIT, ELE_NUM_PER_C0) / ELE_NUM_PER_C0;

        for(uint32_t w_i=0; w_i < repeat_num_n; w_i++){
            uint32_t src_n_offset = w_i * W_UNIT;
            uint32_t ub_src_n_offset = src_n_offset * 1;
            uint32_t dst_m_offset = w_i * W_UNIT;
            uint32_t ub_dst_m_offset = dst_m_offset * transed_n_round;

            uint32_t src_temp_workspace_offset = 0;

            /*
                template <typename T, typename U, bool isSetMask = true, 
                    typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type = true>
                __aicore__ inline void Adds(const LocalTensor<T>& dstLocal, 
                    const LocalTensor<T>& srcLocal, 
                    const U& scalarValue, 
                    uint64_t mask, 
                    const uint8_t repeatTimes, 
                    const UnaryRepeatParams& repeatParams)
            */

            AscendC::Adds<ElementA, ElementA>(temp_workspace[src_temp_workspace_offset],
                    srcTensor_m[ub_src_n_offset],
                    (ElementA)to_temp_scalar,
                    (uint64_t)temp_mask, temp_repeat_times,
                    cache_params);
                
            AscendC::PipeBarrier<PIPE_V>();
            
            for(uint32_t h_i=0; h_i < repeat_num_m; h_i++){

                uint32_t src_m_offset = h_i * H_UNIT;
                uint32_t ub_src_m_offset = src_m_offset * raw_n_round;
                uint32_t dst_n_offset = h_i * H_UNIT;
                uint32_t ub_dst_n_offset = dst_n_offset * 1;

                uint32_t ub_src_offset = ub_src_m_offset + ub_src_n_offset;
                uint32_t ub_dst_offset = ub_dst_m_offset + ub_dst_n_offset;

                
                uint32_t dst_temp_workspace_offset = h_i * H_UNIT * W_UNIT;

                // step 1: transpose source matrix tile to temp workspace

                /*
                template <typename T>
                __aicore__ inline void Transpose(const LocalTensor<T>& dstLocal, 
                    const LocalTensor<T>& srcLocal)
                */
                
                AscendC::Transpose<ElementA>(
                    temp_workspace[dst_temp_workspace_offset],
                    temp_workspace[dst_temp_workspace_offset]
                );
            }
            
            AscendC::PipeBarrier<PIPE_V>();
            // 将局部转换完成的数据归约为行主序即可
            
            for(uint32_t f_k=0; f_k <W_UNIT; f_k++){
                uint32_t final_dst_offset = ub_dst_m_offset + f_k * transed_n_round;
                uint32_t final_workspace_offset = f_k * H_UNIT;
                AscendC::Adds<ElementY, ElementY>(
                    dstTensor[final_dst_offset],
                    temp_workspace[final_workspace_offset],
                    (ElementA)to_temp_scalar,
                    (uint64_t)final_mask,
                    final_repeat_times,
                    final_transed_params
                );
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
};

template <>
struct TileMatrixTranspose<Arch::AtlasA2,
                Gemm::GemmType<half, layout::RowMajor>,
                Gemm::GemmType<float, layout::RowMajor>,
                void>
{
    static constexpr uint32_t H_UNIT = 16;
    static constexpr uint32_t W_UNIT = 16;

    using ElementA = half;
    using ElementX = half;
    using ElementY = float;

    using ElementAccumulator = ElementY;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);
    static constexpr uint32_t OUT_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);

    // Methods

    CATLASS_DEVICE
    TileMatrixTranspose() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        AscendC::LocalTensor<ElementA> temp_workspace,
        LayoutDst const &layoutDst, 
        LayoutSrc const &layoutSrc
    )
    {
        uint32_t raw_m_actual = layoutSrc.shape(0);
        uint32_t raw_n_actual = layoutSrc.shape(1);
        // RoundUp(n_round, repeat_size)
        raw_m_actual = RoundUp(raw_m_actual, H_UNIT);
        raw_n_actual = RoundUp(raw_n_actual, W_UNIT);

        uint32_t transed_m_actual = raw_n_actual;
        uint32_t transed_n_actual = raw_m_actual;

        uint32_t raw_m_round = layoutDst.shape(1);
        uint32_t raw_n_round = layoutDst.shape(0);

        uint32_t transed_m_round = raw_n_round;
        uint32_t transed_n_round = raw_m_round;

        uint32_t repeat_num_m = raw_m_actual / H_UNIT;
        uint32_t repeat_num_n = raw_n_actual / W_UNIT;

        float to_temp_scalar = 0.0f;

        uint32_t temp_repeat_times = raw_m_actual;
        uint32_t temp_mask = W_UNIT;
        AscendC::UnaryRepeatParams cache_params;
        cache_params.dstBlkStride = 1;
        cache_params.srcBlkStride = 1;
        cache_params.dstRepStride = RoundUp(W_UNIT, ELE_NUM_PER_C0) / ELE_NUM_PER_C0;
        cache_params.srcRepStride = RoundUp(raw_n_round, ELE_NUM_PER_C0) / ELE_NUM_PER_C0;

        uint32_t final_mask= H_UNIT;
        uint32_t final_repeat_times = repeat_num_m;

        AscendC::UnaryRepeatParams final_transed_params;
        final_transed_params.dstBlkStride = 1;
        final_transed_params.srcBlkStride = 1;
        final_transed_params.dstRepStride = RoundUp(H_UNIT, OUT_ELE_NUM_PER_C0) / OUT_ELE_NUM_PER_C0;
        final_transed_params.srcRepStride = RoundUp(H_UNIT * W_UNIT, ELE_NUM_PER_C0) / ELE_NUM_PER_C0;

        for(uint32_t w_i=0; w_i < repeat_num_n; w_i++){
            uint32_t src_n_offset = w_i * W_UNIT;
            uint32_t ub_src_n_offset = src_n_offset * 1;
            uint32_t dst_m_offset = w_i * W_UNIT;
            uint32_t ub_dst_m_offset = dst_m_offset * transed_n_round;

            uint32_t src_temp_workspace_offset = 0;

            /*
                template <typename T, typename U, bool isSetMask = true, 
                    typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type = true>
                __aicore__ inline void Adds(const LocalTensor<T>& dstLocal, 
                    const LocalTensor<T>& srcLocal, 
                    const U& scalarValue, 
                    uint64_t mask, 
                    const uint8_t repeatTimes, 
                    const UnaryRepeatParams& repeatParams)
            */

            AscendC::Adds<ElementA, ElementA>(temp_workspace[src_temp_workspace_offset],
                    srcTensor_m[ub_src_n_offset],
                    (ElementA)to_temp_scalar,
                    (uint64_t)temp_mask, temp_repeat_times,
                    cache_params);
                
            AscendC::PipeBarrier<PIPE_V>();
            
            for(uint32_t h_i=0; h_i < repeat_num_m; h_i++){

                uint32_t src_m_offset = h_i * H_UNIT;
                uint32_t ub_src_m_offset = src_m_offset * raw_n_round;
                uint32_t dst_n_offset = h_i * H_UNIT;
                uint32_t ub_dst_n_offset = dst_n_offset * 1;

                uint32_t ub_src_offset = ub_src_m_offset + ub_src_n_offset;
                uint32_t ub_dst_offset = ub_dst_m_offset + ub_dst_n_offset;

                
                uint32_t dst_temp_workspace_offset = h_i * H_UNIT * W_UNIT;

                // step 1: transpose source matrix tile to temp workspace

                /*
                template <typename T>
                __aicore__ inline void Transpose(const LocalTensor<T>& dstLocal, 
                    const LocalTensor<T>& srcLocal)
                */
                
                AscendC::Transpose<ElementA>(
                    temp_workspace[dst_temp_workspace_offset],
                    temp_workspace[dst_temp_workspace_offset]
                );
            }
            
            AscendC::PipeBarrier<PIPE_V>();
            // 将局部转换完成的数据归约为行主序即可
            
            for(uint32_t f_k=0; f_k <W_UNIT; f_k++){
                uint32_t final_dst_offset = ub_dst_m_offset + f_k * transed_n_round;
                uint32_t final_workspace_offset = f_k * H_UNIT;
                // AscendC::Adds<ElementY, ElementY>(
                //     dstTensor[final_dst_offset],
                //     temp_workspace[final_workspace_offset],
                //     (ElementA)to_temp_scalar,
                //     final_mask,
                //     final_repeat_times,
                //     final_transed_params
                // );
                /*
                template <typename T1, typename T2, bool isSetMask = true>
                __aicore__ inline void Cast(const LocalTensor<T1>& dstLocal, 
                    const LocalTensor<T2>& srcLocal, 
                    const RoundMode& round_mode, 
                    const uint64_t mask, 
                    const uint8_t repeatTimes, 
                    const UnaryRepeatParams& repeatParams)
                */
                AscendC::Cast<ElementY, ElementA, true>(
                    dstTensor[final_dst_offset],
                    temp_workspace[final_workspace_offset],
                    AscendC::RoundMode::CAST_NONE,
                    (uint64_t)final_mask,
                    final_repeat_times,
                    final_transed_params);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
};

template <>
struct TileMatrixTranspose<Arch::AtlasA2,
                Gemm::GemmType<float, layout::RowMajor>,
                Gemm::GemmType<float, layout::RowMajor>,
                void>
{
    using ElementA = float;
    using ElementX = float;
    using ElementY = float;

    static constexpr uint32_t H_UNIT = 16;
    static constexpr uint32_t W_UNIT = 16;
    static constexpr uint32_t C_UNIT_LOWER_LIMIT = BYTE_PER_C0 / sizeof(ElementY);
    static constexpr uint32_t C_UNIT_UPPER_LIMIT =  24;


    using ElementAccumulator = ElementY;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);
    static constexpr uint32_t OUT_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);
    static constexpr uint32_t ELEM_NUM_LIMIT = 32 *1024 / sizeof(ElementA);
    // static constexpr uint32_t ELEM_NUM_LIMIT_PER_HW = (16 *1024 / sizeof(ElementA)) / C_UNIT;

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        AscendC::LocalTensor<uint8_t> sharedTmpBuffer,
        LayoutDst const &layoutDst, 
        LayoutSrc const &layoutSrc
    )
    {
        /*
        template <typename T>
        __aicore__ inline void Transpose(const LocalTensor<T>& dstLocal, 
            const LocalTensor<T> &srcLocal, 
            const LocalTensor<uint8_t> &sharedTmpBuffer, 
            const TransposeParamsExt &transposeParams)
        */

        uint32_t raw_m_actual = layoutSrc.shape(0);
        uint32_t raw_n_actual = layoutSrc.shape(1);
        // RoundUp(n_round, repeat_size)

        raw_m_actual = RoundUp(raw_m_actual, ELE_NUM_PER_C0);
        raw_m_actual = RoundUp(raw_m_actual, H_UNIT);

        uint32_t transed_m_actual = raw_n_actual;
        uint32_t transed_n_actual = raw_m_actual;

        uint32_t raw_m_round = layoutDst.shape(1);
        uint32_t raw_n_round = layoutDst.shape(0);

        uint32_t transed_m_round = raw_n_round;
        uint32_t transed_n_round = raw_m_round;

        uint32_t C_UNIT_RAW = raw_n_round;
        uint32_t C_UNIT = (C_UNIT_RAW > C_UNIT_UPPER_LIMIT) ? C_UNIT_UPPER_LIMIT : C_UNIT_RAW;
        C_UNIT = (C_UNIT < C_UNIT_LOWER_LIMIT) ? C_UNIT_LOWER_LIMIT: C_UNIT;
        // C_UNIT = (C_UNIT < 1) ? 1 : C_UNIT;

        uint32_t repeat_num_n = 1;
        uint32_t repeat_num_m = 1;

        // uint32_t repeat_num_m = (raw_m_actual + ELEM_NUM_LIMIT_PER_HW - 1) / ELEM_NUM_LIMIT_PER_HW;
        // uint32_t repeat_num_n = (raw_n_actual + C_UNIT - 1) / C_UNIT;

        float to_temp_scalar = 0.0f;
        
        AscendC::UnaryRepeatParams cache_params;
        cache_params.dstBlkStride = 1;
        cache_params.srcBlkStride = 1;
        cache_params.dstRepStride = RoundUp(C_UNIT, ELE_NUM_PER_C0) / ELE_NUM_PER_C0;
        cache_params.srcRepStride = RoundUp(raw_n_round, ELE_NUM_PER_C0) / ELE_NUM_PER_C0;

        AscendC::TransposeType transposeType = AscendC::TransposeType::TRANSPOSE_NHWC2NCHW;

        AscendC::TransposeParamsExt transposeParams;
        transposeParams.nSize = uint16_t(1);
        transposeParams.cSize = (uint16_t)C_UNIT;

        uint32_t temp_w_size = ((raw_m_actual + H_UNIT - 1) / H_UNIT);
        // temp_w_size = (temp_w_size < ELE_NUM_PER_C0) ? ELE_NUM_PER_C0 : temp_w_size;
        // uint32_t temp_h_size = raw_m_actual / temp_w_size;

        transposeParams.hSize = (uint16_t)H_UNIT;
        transposeParams.wSize = (uint16_t)temp_w_size;
        transposeParams.transposeType = transposeType;

        AscendC::Transpose<ElementA>(dstTensor,
                srcTensor_m,
                sharedTmpBuffer,
                transposeParams);

        
        // uint32_t src_m_offset = 0;
        // uint32_t ub_src_m_offset = src_m_offset * raw_n_round;
        // uint32_t dst_n_offset = 0;
        // uint32_t ub_dst_n_offset = dst_n_offset * 1;  
    
        // for(uint32_t w_i=0; w_i < repeat_num_n; w_i++){
        //     uint32_t temp_repeat_times = raw_m_actual;
        //     uint32_t temp_mask = (w_i == (repeat_num_n - 1)) ? (raw_n_actual - C_UNIT * w_i) : C_UNIT;

        //     uint32_t C_UNIT_EFFECTIVE = RoundUp(temp_mask, ELE_NUM_PER_C0);

        //     cache_params.dstRepStride = C_UNIT_EFFECTIVE / ELE_NUM_PER_C0;
        //     cache_params.srcRepStride = RoundUp(raw_n_round, ELE_NUM_PER_C0) / ELE_NUM_PER_C0;

        //     transposeParams.cSize = (uint16_t)C_UNIT_EFFECTIVE;

        //     uint32_t src_n_offset = C_UNIT * w_i;
        //     uint32_t ub_src_n_offset = src_n_offset * 1;

        //     uint32_t dst_m_offset = C_UNIT * w_i;
        //     uint32_t ub_dst_m_offset = dst_m_offset * transed_n_actual;

        //     uint32_t src_temp_workspace_offset = 0;

        //     uint32_t ub_src_offset = ub_src_m_offset + ub_src_n_offset;
        //     uint32_t ub_dst_offset = ub_dst_m_offset + ub_dst_n_offset;

        //     AscendC::Adds<ElementA, ElementA>(temp_workspace[src_temp_workspace_offset],
        //         srcTensor_m[ub_src_n_offset],
        //         (ElementA)to_temp_scalar,
        //         (uint64_t)temp_mask, temp_repeat_times,
        //         cache_params);
                
        //     AscendC::PipeBarrier<PIPE_V>();

        //     /*
        //         template <typename T>
        //         __aicore__ inline void Transpose(const LocalTensor<T>& dstLocal, 
        //             const LocalTensor<T> &srcLocal, 
        //             const LocalTensor<uint8_t> &sharedTmpBuffer, 
        //             const TransposeParamsExt &transposeParams)
        //      */

            

            // AscendC::PipeBarrier<PIPE_V>();
            /*
                表2 接口参数说明
                参数名称                输入/输出               含义
                dstLocal                输出                目的操作数。
                                                        类型为LocalTensor，
                                                        支持的TPosition为VECIN/VECCALC/VECOUT。
                                                        LocalTensor的起始地址需要32字节对齐。

                srcLocal                输入                源操作数。
                                                        类型为LocalTensor，
                                                        支持的TPosition为VECIN/VECCALC/VECOUT。
                                                        LocalTensor的起始地址需要32字节对齐。
                                                        数据类型需要与dstLocal保持一致。

                sharedTmpBuffer         输入            共享的临时Buffer，
                                                        sharedTmpBuffer的大小参考表4。

                transposeParams         输入            控制Transpose的数据结构。
                                                        结构体内包含：
                                                        输入的shape信息和transposeType参数。
                                                        该数据结构的定义请参考表3。

                struct TransposeParamsExt {
                    __aicore__ TransposeParamsExt() {}
                    __aicore__ TransposeParamsExt(const uint16_t nSizeIn, const uint16_t cSizeIn, const uint16_t hSizeIn,
                        const uint16_t wSizeIn, const TransposeType transposeTypeIn) : 
                        nSize(nSizeIn), cSize(cSizeIn), 
                        hSize(hSizeIn), wSize(wSizeIn),
                        transposeType(transposeTypeIn)
                        {}

                    uint16_t nSize = 0;
                    uint16_t cSize = 0;
                    uint16_t hSize = 0;
                    uint16_t wSize = 0;
                    TransposeType transposeType = TransposeType::TRANSPOSE_ND2ND_B16;
            */

            /*
                        表3 TransposeParamsExt结构体内参数说明
                参数名称            含义
                nSize           n轴长度。默认值为0。
                                二维矩阵数据块转置，无需传入，传入数值无效。
                                [N,C,H,W]与[N,H,W,C]数据格式互相转换，
                                取值范围：nSize∈[0, 65535]。
                
                cSize           c轴长度。默认值为0。
                                二维矩阵数据块转置，无需传入，传入数值无效。
                                [N,C,H,W]与[N,H,W,C]数据格式互相转换，
                                取值范围：cSize∈[0, 4095]
                
                hSize           h轴长度。默认值为0。  
                                二维矩阵数据块转置，固定传入16。
                                [N,C,H,W]与[N,H,W,C]数据格式互相转换，
                                取值范围：hSize * wSize ∈[0, 4095]，
                                hSize * wSize * sizeof(T)需要保证32B对齐。

                wSize           w轴长度。默认值为0。
                                二维矩阵数据块转置，固定传入16。
                                [N,C,H,W]与[N,H,W,C]数据格式互相转换，
                                取值范围：hSize * wSize ∈[0, 4095]，
                                hSize * wSize * sizeof(T)需要保证32B对齐。

                transposeType    数据排布及reshape的类型，类型为TransposeType枚举类。
                                默认值为TRANSPOSE_ND2ND_B16。

                enum class TransposeType : uint8_t {
                    TRANSPOSE_TYPE_NONE,           // API不做任何处理
                    TRANSPOSE_ND2ND_B16,           // [16,16]二维矩阵转置
                    TRANSPOSE_NCHW2NHWC,           // [N,C,H,W]->[N,H,W,C]，
                    TRANSPOSE_NHWC2NCHW            // [N,H,W,C]->[N,C,H,W]
            */

            /*
                    表4 增强转置接口sharedTmpBuffer所需的大小
                transposeType               sharedTmpBuffer所需的大小

                TRANSPOSE_ND2ND_B16         不需要临时Buffer。
                
                TRANSPOSE_NCHW2NHWC         针对以下型号：
                                            Atlas 推理系列产品AI Core
                                            不需要临时Buffer。

                                            针对以下型号：
                                            Atlas A2 训练系列产品/Atlas 800I A2 推理产品/
                                            A200I A2 Box 异构组件
                                            Atlas A3 训练系列产品/Atlas A3 推理系列产品
                                            临时Buffer的大小按照下述计算规则（伪代码）进行计算。

                                            auto h0 = 16; // 当数据类型的位宽为8时，
                                            // h0 = 32；其他情况下，h0 = 16
                                            auto w0 = 32 / sizeof(type);  
                                            // type代表数据类型
                                            auto tmpBufferSize = (cSize + 2)  * h0 * w0 * sizeof(type);
                
                TRANSPOSE_NHWC2NCHW         针对以下型号：
                                            Atlas 推理系列产品AI Core
                                            不需要临时Buffer。

                                            针对以下型号：
                                            Atlas A2 训练系列产品/Atlas 800I A2 推理产品/
                                            A200I A2 Box 异构组件
                                            Atlas A3 训练系列产品/Atlas A3 推理系列产品
                                            临时Buffer的大小按照下述计算规则（伪代码）进行计算。

                                            auto h0 = 16; // 当数据类型的位宽为8时，h0 = 32；
                                            // 其他情况下，h0 = 16
                                            auto w0 = 32 / sizeof(type); 
                                            // type代表数据类型
                                            auto tmpBufferSize = (cSize  * 2 + 1)  * h0 * w0 * sizeof(type);
            */

        // }
    }
};

}
#endif // CATLASS_GEMV_TILE_TILE_MATRIX_TRANSPOSE_HPP_SELF

