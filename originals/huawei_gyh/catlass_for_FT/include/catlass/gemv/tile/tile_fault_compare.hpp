#ifndef CATLASS_GEMV_TILE_TILE_FAULT_VCOMP_HPP
#define CATLASS_GEMV_TILE_TILE_FAULT_VCOMP_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemv/helper.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/gemm_type.hpp"

namespace Catlass::Gemv::Tile {
template <
    /// Tag indicating architecture
    Gemv::helper::FT_COMP_TYPE COMP_TYPE_,
    class ArchTag,
    class ZType,
    class XType,
    class YType
>
struct TileFaultVcompare
{
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported TileFaultVmad, can not find the specialization.");
};

template<
    class ElementZ,
    class ElementX,
    class ElementY
>
struct TileFaultVcompare<
        Gemv::helper::FT_COMP_TYPE::XOR,
        Arch::AtlasA2, 
        Gemm::GemmType<ElementZ, Catlass::layout::VectorLayout>,
        Gemm::GemmType<ElementX, Catlass::layout::VectorLayout>,
        Gemm::GemmType<ElementY, Catlass::layout::VectorLayout>
        > 
{
    using ElementWIn = uint16_t;
    using ElementWTmp = int32_t;
    using ElementShuffle = uint8_t;

    using LayoutDst = Catlass::layout::VectorLayout;
    using LayoutSrc = Catlass::layout::VectorLayout;

    static constexpr uint32_t ELE_NUM_PER_BLK_IN = BYTE_PER_BLK / sizeof(ElementWIn);
    static constexpr uint32_t ELE_NUM_PER_BLK_TMP = BYTE_PER_BLK / sizeof(ElementWTmp);
    static constexpr uint32_t ELE_NUM_PER_BLK_OUT = BYTE_PER_BLK / sizeof(ElementZ);

    // Mehtods

    CATLASS_DEVICE
    TileFaultVcompare() {};

    // AscendC::LocalTensor<ElementWIn> workSpaceTensor,
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementZ> dstTensor,
        AscendC::LocalTensor<ElementX> srcTensor_x,
        AscendC::LocalTensor<ElementY> srcTensor_y,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc, ElementX threshold
    ){

        uint32_t n_actual = layoutSrc.shape(0);
        uint32_t n_round_xor = n_actual * sizeof(ElementX) / sizeof(ElementWIn);
        uint32_t ELE_NUM_PER_REPEAT_IN = ELE_NUM_PER_BLK_IN * 8;
        uint32_t ELE_NUM_PER_REPEAT_TMP = ELE_NUM_PER_BLK_TMP * 8;

        n_round_xor = RoundUp(n_round_xor, ELE_NUM_PER_REPEAT_IN);

        AscendC::LocalTensor<ElementWIn> srcTensor_x_rep = srcTensor_x.template ReinterpretCast<ElementWIn>();
        AscendC::LocalTensor<ElementWIn> srcTensor_y_rep = srcTensor_y.template ReinterpretCast<ElementWIn>();
        AscendC::LocalTensor<ElementShuffle> sharedTmpBuffer = dstTensor.template ReinterpretCast<ElementShuffle>();

        // AscendC::Xor(workSpaceTensor, srcTensor_x_rep, srcTensor_y_rep, sharedTmpBuffer, n_round_xor);
        AscendC::Xor(srcTensor_x_rep, srcTensor_x_rep, srcTensor_y_rep, sharedTmpBuffer, n_round_xor);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::LocalTensor<ElementWTmp> srcTensor_x_rep_int = srcTensor_x_rep.template ReinterpretCast<ElementWTmp>();
        
        // uint64_t mask_num = BYTE_PER_BLK * 8 / sizeof(ElementWTmp);

        uint32_t n_round_comp = n_round_xor * sizeof(ElementWIn) / sizeof(ElementWTmp);
        n_round_comp = RoundUp(n_round_comp, ELE_NUM_PER_REPEAT_TMP);
        
        AscendC::CompareScalar(dstTensor, srcTensor_x_rep_int, 
            static_cast<ElementWTmp>(0), AscendC::CMPMODE::EQ, n_round_comp);
        
        // AscendC::PipeBarrier<PIPE_V>();
    }
};

template<
    class ElementZ,
    class ElementX,
    class ElementY
>
struct TileFaultVcompare<
        Gemv::helper::FT_COMP_TYPE::COMPARE,
        Arch::AtlasA2, 
        Gemm::GemmType<ElementZ, Catlass::layout::VectorLayout>,
        Gemm::GemmType<ElementX, Catlass::layout::VectorLayout>,
        Gemm::GemmType<ElementY, Catlass::layout::VectorLayout>
        > 
{
    using ElementWIn = int32_t;
    using ElementWTmp = int32_t;
    // using ElementShuffle = uint8_t;

    using LayoutDst = Catlass::layout::VectorLayout;
    using LayoutSrc = Catlass::layout::VectorLayout;

    static constexpr uint32_t ELE_NUM_PER_BLK_IN = BYTE_PER_BLK / sizeof(ElementWIn);
    static constexpr uint32_t ELE_NUM_PER_BLK_TMP = BYTE_PER_BLK / sizeof(ElementWTmp);
    static constexpr uint32_t ELE_NUM_PER_BLK_OUT = BYTE_PER_BLK / sizeof(ElementZ);

    // Mehtods

    CATLASS_DEVICE
    TileFaultVcompare() {};

    // AscendC::LocalTensor<ElementWIn> workSpaceTensor,
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementZ> dstTensor,
        AscendC::LocalTensor<ElementX> srcTensor_x,
        AscendC::LocalTensor<ElementY> srcTensor_y,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc, ElementX threshold
    ){

        uint32_t n_actual = layoutSrc.shape(0);
        uint32_t n_round_comp = n_actual * sizeof(ElementX) / sizeof(ElementWIn);
        uint32_t ELE_NUM_PER_REPEAT_IN = ELE_NUM_PER_BLK_IN * 8;
        uint32_t ELE_NUM_PER_REPEAT_TMP = ELE_NUM_PER_BLK_TMP * 8;

        n_round_comp = RoundUp(n_round_comp, ELE_NUM_PER_REPEAT_IN);

        AscendC::LocalTensor<ElementWIn> srcTensor_x_rep = srcTensor_x.template ReinterpretCast<ElementWIn>();
        AscendC::LocalTensor<ElementWIn> srcTensor_y_rep = srcTensor_y.template ReinterpretCast<ElementWIn>();
        
        AscendC::Compare(dstTensor, srcTensor_x_rep, 
            srcTensor_y_rep, AscendC::CMPMODE::EQ, n_round_comp);
        // AscendC::PipeBarrier<PIPE_V>();
    }
};

template<
    class ElementZ,
    class ElementX,
    class ElementY
>
struct TileFaultVcompare<
        Gemv::helper::FT_COMP_TYPE::SUB,
        Arch::AtlasA2, 
        Gemm::GemmType<ElementZ, Catlass::layout::VectorLayout>,
        Gemm::GemmType<ElementX, Catlass::layout::VectorLayout>,
        Gemm::GemmType<ElementY, Catlass::layout::VectorLayout>
        > 
{
    using ElementWIn = ElementX;
    using ElementWTmp = ElementX;
    // using ElementShuffle = uint8_t;

    using LayoutDst = Catlass::layout::VectorLayout;
    using LayoutSrc = Catlass::layout::VectorLayout;

    static constexpr uint32_t ELE_NUM_PER_BLK_IN = BYTE_PER_BLK / sizeof(ElementWIn);
    static constexpr uint32_t ELE_NUM_PER_BLK_TMP = BYTE_PER_BLK / sizeof(ElementWTmp);
    static constexpr uint32_t ELE_NUM_PER_BLK_OUT = BYTE_PER_BLK / sizeof(ElementZ);

    // Mehtods

    CATLASS_DEVICE
    TileFaultVcompare() {};

    // AscendC::LocalTensor<ElementWIn> workSpaceTensor,
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementZ> dstTensor,
        AscendC::LocalTensor<ElementX> srcTensor_x,
        AscendC::LocalTensor<ElementY> srcTensor_y,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc, ElementX threshold
    ){

        uint32_t n_actual = layoutSrc.shape(0);
        uint32_t n_round_sub = n_actual * sizeof(ElementX) / sizeof(ElementWIn);
        uint32_t ELE_NUM_PER_REPEAT_IN = ELE_NUM_PER_BLK_IN * 8;
        uint32_t ELE_NUM_PER_REPEAT_TMP = ELE_NUM_PER_BLK_TMP * 8;

        n_round_sub = RoundUp(n_round_sub, ELE_NUM_PER_REPEAT_IN);
        
        uint64_t mask = ELE_NUM_PER_REPEAT_TMP;

        AscendC::BinaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 1;
        repeatParams.dstRepStride = 8;
        repeatParams.src0RepStride = 8;
        repeatParams.src1RepStride = 8;
        
        uint32_t repeatTimes = n_round_sub / ELE_NUM_PER_BLK_TMP;

        // AscendC::Sub(workSpaceTensor, srcTensor_x, srcTensor_y, mask, (uint8_t)repeatTimes, repeatParams);
        // mask, (uint8_t)repeatTimes, repeatParams
        AscendC::Sub(srcTensor_x, srcTensor_x, srcTensor_y,n_round_sub);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Abs(srcTensor_x, srcTensor_x, n_round_sub);

        AscendC::PipeBarrier<PIPE_V>();

        uint32_t n_round_comp = n_round_sub * sizeof(ElementWIn) / sizeof(ElementWTmp);
        n_round_comp = RoundUp(n_round_comp, ELE_NUM_PER_REPEAT_TMP);
        
        AscendC::CompareScalar(dstTensor, srcTensor_x, 
            threshold, AscendC::CMPMODE::LE, n_round_comp);
        
        // AscendC::PipeBarrier<PIPE_V>();
    }
};


template<
    class ElementZ,
    class ElementX,
    class ElementY
>
struct TileFaultVcompare<
        Gemv::helper::FT_COMP_TYPE::RSUB,
        Arch::AtlasA2, 
        Gemm::GemmType<ElementZ, Catlass::layout::VectorLayout>,
        Gemm::GemmType<ElementX, Catlass::layout::VectorLayout>,
        Gemm::GemmType<ElementY, Catlass::layout::VectorLayout>
        > 
{
    using ElementWIn = ElementX;
    using ElementWTmp = ElementX;
    // using ElementShuffle = uint8_t;

    using LayoutDst = Catlass::layout::VectorLayout;
    using LayoutSrc = Catlass::layout::VectorLayout;

    static constexpr uint32_t ELE_NUM_PER_BLK_IN = BYTE_PER_BLK / sizeof(ElementWIn);
    static constexpr uint32_t ELE_NUM_PER_BLK_TMP = BYTE_PER_BLK / sizeof(ElementWTmp);
    static constexpr uint32_t ELE_NUM_PER_BLK_OUT = BYTE_PER_BLK / sizeof(ElementZ);

    // Mehtods

    CATLASS_DEVICE
    TileFaultVcompare() {};

    // AscendC::LocalTensor<ElementWIn> workSpaceTensor,
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementZ> dstTensor,
        AscendC::LocalTensor<ElementX> srcTensor_x,
        AscendC::LocalTensor<ElementY> srcTensor_y,
        AscendC::LocalTensor<ElementX> srcTensor_thre,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc, ElementX threshold
    ){

        uint32_t n_actual = layoutSrc.shape(0);
        uint32_t n_round_sub = n_actual * sizeof(ElementX) / sizeof(ElementWIn);
        uint32_t ELE_NUM_PER_REPEAT_IN = ELE_NUM_PER_BLK_IN * 8;
        uint32_t ELE_NUM_PER_REPEAT_TMP = ELE_NUM_PER_BLK_TMP * 8;

        ElementWTmp threshold_row = static_cast<ElementWTmp>(0.0f);

        n_round_sub = RoundUp(n_round_sub, ELE_NUM_PER_REPEAT_IN);
        
        uint64_t mask = ELE_NUM_PER_REPEAT_TMP;

        AscendC::BinaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 1;
        repeatParams.dstRepStride = 8;
        repeatParams.src0RepStride = 8;
        repeatParams.src1RepStride = 8;
        
        uint32_t repeatTimes = n_round_sub / ELE_NUM_PER_BLK_TMP;

        // AscendC::Sub(workSpaceTensor, srcTensor_x, srcTensor_y, mask, (uint8_t)repeatTimes, repeatParams);
        // mask, (uint8_t)repeatTimes, repeatParams
        AscendC::Sub(srcTensor_x, srcTensor_x, srcTensor_y,n_round_sub);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Abs(srcTensor_x, srcTensor_x, n_round_sub);

        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Sub(srcTensor_x, srcTensor_x, srcTensor_thre, n_round_sub);

        AscendC::PipeBarrier<PIPE_V>();

        uint32_t n_round_comp = n_round_sub * sizeof(ElementWIn) / sizeof(ElementWTmp);
        n_round_comp = RoundUp(n_round_comp, ELE_NUM_PER_REPEAT_TMP);
        
        AscendC::CompareScalar(dstTensor, srcTensor_x, 
            threshold_row, AscendC::CMPMODE::LE, n_round_comp);
        
        // AscendC::PipeBarrier<PIPE_V>();
    }
};


// 将比较进行矩阵化
template<
    class ElementZ,
    class ElementX,
    class ElementY
>
struct TileFaultVcompare<
        Gemv::helper::FT_COMP_TYPE::RSUB,
        Arch::AtlasA2, 
        Gemm::GemmType<ElementZ, Catlass::layout::RowMajor>,
        Gemm::GemmType<ElementX, Catlass::layout::RowMajor>,
        Gemm::GemmType<ElementY, Catlass::layout::RowMajor>
        > 
{
    using ElementWIn = ElementX;
    using ElementWTmp = ElementX;
    // using ElementShuffle = uint8_t;

    using LayoutDst = Catlass::layout::RowMajor;
    using LayoutSrc = Catlass::layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_BLK_IN = BYTE_PER_BLK / sizeof(ElementWIn);
    static constexpr uint32_t ELE_NUM_PER_BLK_TMP = BYTE_PER_BLK / sizeof(ElementWTmp);
    static constexpr uint32_t ELE_NUM_PER_BLK_OUT = BYTE_PER_BLK / sizeof(ElementZ);

    // Mehtods

    CATLASS_DEVICE
    TileFaultVcompare() {};

    // AscendC::LocalTensor<ElementWIn> workSpaceTensor,
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementZ> dstTensor,
        AscendC::LocalTensor<ElementX> srcTensor_x,
        AscendC::LocalTensor<ElementY> srcTensor_y,
        AscendC::LocalTensor<ElementX> srcTensor_thre,
        LayoutDst const &layoutDst, LayoutDst const &layoutDstZ,
        LayoutSrc const &layoutSrc, ElementX threshold
    ){

        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);
        uint32_t n_round_for_z = layoutDstZ.shape(1);

        uint32_t n_round_sub = n_round * sizeof(ElementX) / sizeof(ElementWIn);
        uint32_t ELE_NUM_PER_REPEAT_IN = ELE_NUM_PER_BLK_IN * 8;
        uint32_t ELE_NUM_PER_REPEAT_TMP = ELE_NUM_PER_BLK_TMP * 8;

        ElementWTmp threshold_row = static_cast<ElementWTmp>(0.0f);

        // n_round_sub = RoundUp(n_round_sub, ELE_NUM_PER_REPEAT_IN);
        
        uint64_t mask = ELE_NUM_PER_REPEAT_TMP;

        uint64_t sub_mask = ELE_NUM_PER_REPEAT_TMP;
        uint64_t abs_mask = ELE_NUM_PER_REPEAT_TMP;

        uint32_t repeat_num = n_actual / ELE_NUM_PER_REPEAT_TMP;
        uint32_t remain = n_actual % ELE_NUM_PER_REPEAT_TMP;

        AscendC::BinaryRepeatParams subParams;
        subParams.dstBlkStride = 1;
        subParams.src0BlkStride = 1;
        subParams.src1BlkStride = 1;
        subParams.dstRepStride = RoundUp(n_round_sub, ELE_NUM_PER_REPEAT_TMP) / ELE_NUM_PER_BLK_TMP;
        subParams.src0RepStride = RoundUp(n_round_sub, ELE_NUM_PER_REPEAT_TMP) / ELE_NUM_PER_BLK_TMP;
        subParams.src1RepStride = RoundUp(n_round_sub, ELE_NUM_PER_REPEAT_TMP) / ELE_NUM_PER_BLK_TMP;

        AscendC::UnaryRepeatParams final_params;

        final_params.dstBlkStride = 1;
        final_params.srcBlkStride = 1;
        final_params.dstRepStride = RoundUp(n_round_sub, ELE_NUM_PER_REPEAT_TMP) / ELE_NUM_PER_BLK_TMP;
        final_params.srcRepStride = RoundUp(n_round_sub, ELE_NUM_PER_REPEAT_TMP) / ELE_NUM_PER_BLK_TMP;

        
        // uint32_t repeatTimes = n_round_sub / ELE_NUM_PER_BLK_TMP;

        // AscendC::Sub(workSpaceTensor, srcTensor_x, srcTensor_y, mask, (uint8_t)repeatTimes, repeatParams);
        // mask, (uint8_t)repeatTimes, repeatParams

        for(uint32_t i=0; i < repeat_num; i++){
            uint32_t offset = i * ELE_NUM_PER_REPEAT_TMP;
            
            // AscendC::Sub(srcTensor_x, srcTensor_x, srcTensor_y,n_round_sub);

            AscendC::Sub<ElementY, true>(
                srcTensor_x[offset], 
                srcTensor_x[offset], 
                srcTensor_y[offset], 
                sub_mask,
                m_actual,
                subParams
            );
        }

        AscendC::PipeBarrier<PIPE_V>();

        if (remain > 0)
        {
            uint32_t offset = repeat_num * ELE_NUM_PER_REPEAT_TMP;
            if (offset + remain > n_actual)
            {
                remain = n_actual - offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;

            AscendC::Sub<ElementY, true>(
                srcTensor_x[offset], 
                srcTensor_x[offset], 
                srcTensor_y[offset], 
                remain_mask,
                m_actual,
                subParams
            );

            AscendC::PipeBarrier<PIPE_V>();
        }

        for(uint32_t i=0; i < repeat_num; i++){
            uint32_t offset = i * ELE_NUM_PER_REPEAT_TMP;
            
            // AscendC::Abs(srcTensor_x, srcTensor_x, n_round_sub);

            AscendC::Abs<ElementY, true>(
                srcTensor_x[offset], 
                srcTensor_x[offset], 
                abs_mask,
                m_actual,
                final_params
            );
        }

        AscendC::PipeBarrier<PIPE_V>();

        if (remain > 0)
        {
            uint32_t offset = repeat_num * ELE_NUM_PER_REPEAT_TMP;
            if (offset + remain > n_actual)
            {
                remain = n_actual - offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;

            AscendC::Abs<ElementY, true>(
                srcTensor_x[offset], 
                srcTensor_x[offset], 
                remain_mask,
                m_actual,
                final_params
            );

            AscendC::PipeBarrier<PIPE_V>();
        }

        for(uint32_t i=0; i < repeat_num; i++){
            uint32_t offset = i * ELE_NUM_PER_REPEAT_TMP;
            
            // AscendC::Sub(srcTensor_x, srcTensor_x, srcTensor_thre, n_round_sub);

            AscendC::Sub<ElementY, true>(
                srcTensor_x[offset], 
                srcTensor_x[offset], 
                srcTensor_thre[offset], 
                sub_mask,
                m_actual,
                subParams
            );
        }

        AscendC::PipeBarrier<PIPE_V>();

        if (remain > 0)
        {
            uint32_t offset = repeat_num * ELE_NUM_PER_REPEAT_TMP;
            if (offset + remain > n_actual)
            {
                remain = n_actual - offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;

            AscendC::Sub<ElementY, true>(
                srcTensor_x[offset], 
                srcTensor_x[offset], 
                srcTensor_thre[offset], 
                remain_mask,
                m_actual,
                subParams
            );

            AscendC::PipeBarrier<PIPE_V>();
        }

        uint32_t n_round_comp = n_round_sub * sizeof(ElementWIn) / sizeof(ElementWTmp);
        n_round_comp = RoundUp(n_round_comp, ELE_NUM_PER_REPEAT_TMP);

        // uint32_t out_n_round_byte = (n_round_comp + 8 - 1) / 8;
        // uint32_t out_n_round = (out_n_round_byte + sizeof(ElementZ) - 1) / sizeof(ElementZ);

        uint32_t out_n_round = n_round_for_z;
        // RoundUp(out_n_round, ELE_NUM_PER_BLK_OUT);

        for(uint32_t j=0; j < m_actual; j++){
            
            uint32_t row_offset_src = n_round_comp * j;
            uint32_t row_offset_dst = out_n_round * j;

            AscendC::CompareScalar(
                dstTensor[row_offset_dst], 
                srcTensor_x[row_offset_src], 
                threshold_row, 
                AscendC::CMPMODE::LE, 
                n_round_comp);
        }          
        AscendC::PipeBarrier<PIPE_V>();
    }
};
}

#endif // CATLASS_GEMV_TILE_TILE_FAULT_VCOMP_HPP