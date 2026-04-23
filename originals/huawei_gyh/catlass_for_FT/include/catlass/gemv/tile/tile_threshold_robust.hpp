#ifndef CATLASS_GEMV_TILE_TILE_THRESHOLD_MEAN_MAX_STD_FUSED_HPP_ROBUST
#define CATLASS_GEMV_TILE_TILE_THRESHOLD_MEAN_MAX_STD_FUSED_HPP_ROBUST

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemv/helper.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/gemm_type.hpp"

namespace Catlass::Gemv::Tile {
// template <
//     /// Tag indicating architecture
//     class ArchTag,
//     class AType,
//     class XType,
//     class YType,
//     class BiasType = void
// >
// struct TileThreCalc
// {
//     static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported TileThreCalc, can not find the specialization.");
// };

template <
    class ElementA,
    class ElementX,
    class ElementY
>
struct TileThreCalc<Arch::AtlasA2,
                helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
                Gemm::GemmType<ElementA, layout::RowMajor>,
                Gemm::GemmType<ElementX, layout::VectorLayout>,
                Gemm::GemmType<ElementY, layout::VectorLayout>,
                void>
{
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementX, ElementY>::ElementAccumulator;

    using FT_THRESHOLD_ALGORITHM = helper::FT_THRESHOLD_ALGORITHM;

    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST;
    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);

    // Mehtods

    CATLASS_DEVICE
    TileThreCalc() {};

    // 
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementX> srcMeanTensor,
        AscendC::LocalTensor<ElementY> srcStdTensor,
        AscendC::LocalTensor<ElementY> thre_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        ElementY n_ratio_factor, ElementY n_sqrt_ratio_factor,
        ElementY n_square_ratio_factor, ElementY B_slice_meanabs,
        ElementY B_slice_meansquare,
        ElementY B_slice_var, ElementY B_slice_var_square, 
        ElementY e_max)
    {

        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t m_round = layoutDst.shape(0);

        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementX);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = elem_repeat_size;

        uint64_t add_mask = elem_repeat_size;
        uint32_t repeattimes = CeilDiv(m_actual, elem_repeat_size);


        AscendC::Duplicate<ElementY>(
            thre_workspace,
            (ElementY)0.0,
            int32_t(m_actual) * 5
        );

        AscendC::PipeBarrier<PIPE_V>();

        /*
        template <typename T>
        __aicore__ inline void Abs(
            const LocalTensor<T>& dstLocal, 
            const LocalTensor<T>& srcLocal, const int32_t& calCount)
        */

        AscendC::MulAddDst<ElementY, ElementX>(
            thre_workspace[2 * m_actual],
            srcStdTensor,
            srcStdTensor,
            m_actual);

        AscendC::Abs(srcMeanTensor, srcMeanTensor, int32_t(m_actual));
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::UnaryRepeatParams castparams;
        castparams.dstBlkStride = 1;
        castparams.srcBlkStride = 1;
        castparams.dstRepStride = 8;
        castparams.srcRepStride = 4;

        AscendC::Cast<ElementY, ElementX>(
            thre_workspace[4 * m_actual],
            srcMeanTensor,
            AscendC::RoundMode::CAST_NONE, m_actual);
        
        /*
        template <typename T, typename U>
        __aicore__ inline void MulAddDst(const LocalTensor<T>& dstLocal, 
            const LocalTensor<U>& src0Local, 
            const LocalTensor<U>& src1Local, 
            const int32_t& calCount)
        */
        AscendC::MulAddDst<ElementY, ElementX>(
            thre_workspace[3 * m_actual],
            srcMeanTensor,
            srcMeanTensor,
            m_actual);
        
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetMaskCount();
        AscendC::SetVectorMask<ElementY, AscendC::MaskMode::COUNTER>(m_actual);
        AscendC::Muls<ElementY,false>(
            thre_workspace[4 * m_actual],
            thre_workspace[4 * m_actual],
            n_ratio_factor*B_slice_meanabs,
            AscendC::MASK_PLACEHOLDER,
            1,
            AscendC::UnaryRepeatParams{}
        );

        AscendC::Muls<ElementY,false>(
            thre_workspace[3 * m_actual],
            thre_workspace[3 * m_actual],
            16.0f * n_ratio_factor * B_slice_var_square,
            AscendC::MASK_PLACEHOLDER,
            1,
            AscendC::UnaryRepeatParams{}
        );

        AscendC::Muls<ElementY,false>(
            thre_workspace[m_actual*2],
            thre_workspace[m_actual*2],
            16.0f * n_square_ratio_factor * B_slice_meansquare,
            AscendC::MASK_PLACEHOLDER,
            1,
            AscendC::UnaryRepeatParams{}
        );

        AscendC::Muls<ElementY,false>(
            thre_workspace[m_actual],
            srcStdTensor,
            n_sqrt_ratio_factor * B_slice_var,
            AscendC::MASK_PLACEHOLDER,
            1,
            AscendC::UnaryRepeatParams{}
        );
        AscendC::SetMaskNorm();
        AscendC::ResetMask();

        AscendC::PipeBarrier<PIPE_V>();

        add_mask = (m_actual < elem_repeat_size) ? m_actual : elem_repeat_size;

        AscendC::BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;
        params.dstRepStride = 8;
        params.src0RepStride = 8;
        params.src1RepStride = 8;

        // AscendC::Add<ElementY, true>(
        //     thre_workspace,
        //     thre_workspace[3 * m_actual],
        //     thre_workspace[2 * m_actual],
        //     add_mask,
        //     CeilDiv(m_actual, elem_repeat_size),
        //     params);

        AscendC::Add<ElementY>(
            thre_workspace,
            thre_workspace[3 * m_actual],
            thre_workspace[2 * m_actual],
            m_actual);
        
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Abs(thre_workspace, thre_workspace, m_actual);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sqrt(thre_workspace, thre_workspace, m_actual);
        AscendC::PipeBarrier<PIPE_V>();
        
        // AscendC::Add<ElementY, true>(
        //     thre_workspace,
        //     thre_workspace[4 * m_actual],
        //     thre_workspace,
        //     add_mask,
        //     CeilDiv(m_actual, elem_repeat_size),
        //     params);

         AscendC::Add<ElementY>(
            thre_workspace,
            thre_workspace[4 * m_actual],
            thre_workspace,
            m_actual);

        AscendC::PipeBarrier<PIPE_V>();

        // AscendC::Add<ElementY, true>(
        //     dstTensor,
        //     thre_workspace[m_actual],
        //     thre_workspace,
        //     add_mask,
        //     CeilDiv(m_actual, elem_repeat_size),
        //     params);

        AscendC::Add<ElementY>(
            thre_workspace,
            thre_workspace[m_actual],
            thre_workspace,
            m_actual);
        
        AscendC::PipeBarrier<PIPE_V>();


        AscendC::SetMaskCount();
        AscendC::SetVectorMask<ElementY, AscendC::MaskMode::COUNTER>(m_actual);
        AscendC::Muls<ElementY,false>(
            dstTensor,
            thre_workspace,
            e_max,
            AscendC::MASK_PLACEHOLDER,
            1,
            AscendC::UnaryRepeatParams{}
        );
        AscendC::SetMaskNorm();
        AscendC::ResetMask();          
    }
};



/*
    class ElementA,
    class ElementX,
    class ElementY
*/

template <
    class ElementA
>
struct TileThreCalc<Arch::AtlasA2,
                helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
                Gemm::GemmType<ElementA, layout::RowMajor>,
                Gemm::GemmType<float, layout::VectorLayout>,
                Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementX = float;
    using ElementY = float;

    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementX, ElementY>::ElementAccumulator;

    using FT_THRESHOLD_ALGORITHM = helper::FT_THRESHOLD_ALGORITHM;

    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST;
    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);

    // Mehtods

    CATLASS_DEVICE
    TileThreCalc() {};

    // 
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementX> srcMeanTensor,
        AscendC::LocalTensor<ElementY> srcStdTensor,
        AscendC::LocalTensor<ElementY> thre_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        ElementY n_ratio_factor, ElementY n_sqrt_ratio_factor,
        ElementY n_square_ratio_factor, ElementY B_slice_meanabs,
        ElementY B_slice_meansquare,
        ElementY B_slice_var, ElementY B_slice_var_square, 
        ElementY e_max)
    {

        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t m_round = layoutDst.shape(0);

        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementX);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = elem_repeat_size;

        uint64_t add_mask = elem_repeat_size;
        uint32_t repeattimes = CeilDiv(m_actual, elem_repeat_size);


        AscendC::Duplicate<ElementY>(
            thre_workspace,
            (ElementY)0.0,
            int32_t(m_actual) * 5
        );

        AscendC::PipeBarrier<PIPE_V>();

        /*
        template <typename T>
        __aicore__ inline void Abs(
            const LocalTensor<T>& dstLocal, 
            const LocalTensor<T>& srcLocal, const int32_t& calCount)
        */
        AscendC::MulAddDst<ElementY, ElementX>(
            thre_workspace[2 * m_actual],
            srcStdTensor,
            srcStdTensor,
            m_actual);

        AscendC::Abs(srcMeanTensor, srcMeanTensor, int32_t(m_actual));
        AscendC::PipeBarrier<PIPE_V>();
 
        /*
        template <typename T, typename U>
        __aicore__ inline void MulAddDst(const LocalTensor<T>& dstLocal, 
            const LocalTensor<U>& src0Local, 
            const LocalTensor<U>& src1Local, 
            const int32_t& calCount)
        */
        AscendC::MulAddDst<ElementY, ElementX>(
            thre_workspace[3 * m_actual],
            srcMeanTensor,
            srcMeanTensor,
            m_actual);
        
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetMaskCount();
        AscendC::SetVectorMask<ElementY, AscendC::MaskMode::COUNTER>(m_actual);
        AscendC::Muls<ElementY,false>(
            thre_workspace[4 * m_actual],
            srcMeanTensor,
            n_ratio_factor*B_slice_meanabs,
            AscendC::MASK_PLACEHOLDER,
            1,
            AscendC::UnaryRepeatParams{}
        );

        AscendC::Muls<ElementY,false>(
            thre_workspace[3 * m_actual],
            thre_workspace[3 * m_actual],
            16.0f * n_ratio_factor * B_slice_var_square,
            AscendC::MASK_PLACEHOLDER,
            1,
            AscendC::UnaryRepeatParams{}
        );

        AscendC::Muls<ElementY,false>(
            thre_workspace[m_actual * 2],
            thre_workspace[m_actual * 2],
            16.0f * n_square_ratio_factor * B_slice_meansquare,
            AscendC::MASK_PLACEHOLDER,
            1,
            AscendC::UnaryRepeatParams{}
        );

        AscendC::Muls<ElementY,false>(
            thre_workspace[m_actual],
            srcStdTensor,
            n_sqrt_ratio_factor * B_slice_var,
            AscendC::MASK_PLACEHOLDER,
            1,
            AscendC::UnaryRepeatParams{}
        );
        AscendC::SetMaskNorm();
        AscendC::ResetMask();

        AscendC::PipeBarrier<PIPE_V>();

        add_mask = (m_actual < elem_repeat_size) ? m_actual : elem_repeat_size;

        AscendC::BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;
        params.dstRepStride = 8;
        params.src0RepStride = 8;
        params.src1RepStride = 8;

        // AscendC::Add<ElementY, true>(
        //     thre_workspace,
        //     thre_workspace[3 * m_actual],
        //     thre_workspace[2 * m_actual],
        //     add_mask,
        //     CeilDiv(m_actual, elem_repeat_size),
        //     params);
        /*
        template <typename T>
        __aicore__ inline void 
        Add(const LocalTensor<T>& dstLocal, 
            const LocalTensor<T>& src0Local, 
            const LocalTensor<T>& src1Local, 
            const int32_t& calCount)
        */
        AscendC::Add<ElementY>(
                thre_workspace,
                thre_workspace[3 * m_actual],
                thre_workspace[2 * m_actual], 
                m_actual);

        // AscendC::Abs(thre_workspace, thre_workspace, m_actual);
        
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Abs(thre_workspace, thre_workspace, m_actual);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sqrt(thre_workspace, thre_workspace, m_actual);
        AscendC::PipeBarrier<PIPE_V>();
        
        // AscendC::Add<ElementY, true>(
        //     thre_workspace,
        //     thre_workspace[4 * m_actual],
        //     thre_workspace,
        //     add_mask,
        //     CeilDiv(m_actual, elem_repeat_size),
        //     params);
        
        AscendC::Add<ElementY>(
                thre_workspace,
                thre_workspace[4 * m_actual],
                thre_workspace, 
                m_actual);

        AscendC::PipeBarrier<PIPE_V>();

        // AscendC::Add<ElementY, true>(
        //     dstTensor,
        //     thre_workspace[m_actual],
        //     thre_workspace,
        //     add_mask,
        //     CeilDiv(m_actual, elem_repeat_size),
        //     params);

        AscendC::Add<ElementY>(
                thre_workspace,
                thre_workspace[m_actual],
                thre_workspace, 
                m_actual);
        
        AscendC::PipeBarrier<PIPE_V>();


        AscendC::SetMaskCount();
        AscendC::SetVectorMask<ElementY, AscendC::MaskMode::COUNTER>(m_actual);
        AscendC::Muls<ElementY,false>(
            dstTensor,
            thre_workspace,
            e_max,
            AscendC::MASK_PLACEHOLDER,
            1,
            AscendC::UnaryRepeatParams{}
        );
        AscendC::SetMaskNorm();
        AscendC::ResetMask();
        
        AscendC::PipeBarrier<PIPE_V>();
    }
};

template <
    class ElementA
>
struct TileThreCalc<Arch::AtlasA2,
                helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
                Gemm::GemmType<ElementA, layout::RowMajor>,
                Gemm::GemmType<half, layout::VectorLayout>,
                Gemm::GemmType<half, layout::VectorLayout>,
                void>
{
    using ElementX = half;
    using ElementY = half;

    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementX, ElementY>::ElementAccumulator;

    using FT_THRESHOLD_ALGORITHM = helper::FT_THRESHOLD_ALGORITHM;

    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST;
    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);

    // Mehtods

    CATLASS_DEVICE
    TileThreCalc() {};

    // 
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementX> srcMeanTensor,
        AscendC::LocalTensor<ElementY> srcStdTensor,
        AscendC::LocalTensor<ElementY> thre_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        ElementY n_ratio_factor, ElementY n_sqrt_ratio_factor,
        ElementY n_square_ratio_factor, ElementY B_slice_meanabs,
        ElementY B_slice_meansquare,
        ElementY B_slice_var, ElementY B_slice_var_square, 
        ElementY e_max)
    {

        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t m_round = layoutDst.shape(0);

        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementX);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = elem_repeat_size;

        uint64_t add_mask = elem_repeat_size;
        uint32_t repeattimes = CeilDiv(m_actual, elem_repeat_size);


        AscendC::Duplicate<ElementY>(
            thre_workspace,
            (ElementY)0.0,
            int32_t(m_actual) * 5
        );

        AscendC::PipeBarrier<PIPE_V>();

        /*
        template <typename T>
        __aicore__ inline void Abs(
            const LocalTensor<T>& dstLocal, 
            const LocalTensor<T>& srcLocal, const int32_t& calCount)
        */
        AscendC::MulAddDst<ElementY, ElementX>(
            thre_workspace[2 * m_actual],
            srcStdTensor,
            srcStdTensor,
            m_actual);

        AscendC::Abs(srcMeanTensor, srcMeanTensor, int32_t(m_actual));
        AscendC::PipeBarrier<PIPE_V>();
 
        /*
        template <typename T, typename U>
        __aicore__ inline void MulAddDst(const LocalTensor<T>& dstLocal, 
            const LocalTensor<U>& src0Local, 
            const LocalTensor<U>& src1Local, 
            const int32_t& calCount)
        */
        AscendC::MulAddDst<ElementY, ElementX>(
            thre_workspace[3 * m_actual],
            srcMeanTensor,
            srcMeanTensor,
            m_actual);
        
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetMaskCount();
        AscendC::SetVectorMask<ElementY, AscendC::MaskMode::COUNTER>(m_actual);
        AscendC::Muls<ElementY,false>(
            thre_workspace[4 * m_actual],
            srcMeanTensor,
            (ElementY)((float)n_ratio_factor*(float)B_slice_meanabs),
            AscendC::MASK_PLACEHOLDER,
            1,
            AscendC::UnaryRepeatParams{}
        );

        AscendC::Muls<ElementY,false>(
            thre_workspace[3 * m_actual],
            thre_workspace[3 * m_actual],
            (ElementY)(16.0f * (float)n_ratio_factor * (float)B_slice_var_square),
            AscendC::MASK_PLACEHOLDER,
            1,
            AscendC::UnaryRepeatParams{}
        );

        AscendC::Muls<ElementY,false>(
            thre_workspace[m_actual*2],
            thre_workspace[m_actual*2],
            (ElementY)(16.0f * (float)n_square_ratio_factor * (float)B_slice_meansquare),
            AscendC::MASK_PLACEHOLDER,
            1,
            AscendC::UnaryRepeatParams{}
        );

        AscendC::Muls<ElementY,false>(
            thre_workspace[m_actual],
            srcStdTensor,
            (ElementY)((float)n_sqrt_ratio_factor * (float)B_slice_var),
            AscendC::MASK_PLACEHOLDER,
            1,
            AscendC::UnaryRepeatParams{}
        );
        AscendC::SetMaskNorm();
        AscendC::ResetMask();

        AscendC::PipeBarrier<PIPE_V>();

        add_mask = (m_actual < elem_repeat_size) ? m_actual : elem_repeat_size;

        AscendC::BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;
        params.dstRepStride = 8;
        params.src0RepStride = 8;
        params.src1RepStride = 8;

        // AscendC::Add<ElementY, true>(
        //     thre_workspace,
        //     thre_workspace[3 * m_actual],
        //     thre_workspace[2 * m_actual],
        //     add_mask,
        //     CeilDiv(m_actual, elem_repeat_size),
        //     params);
        AscendC::Add<ElementY>(
            thre_workspace,
            thre_workspace[3 * m_actual],
            thre_workspace[2 * m_actual],
            m_actual);
        
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Abs(thre_workspace, thre_workspace, m_actual);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sqrt(thre_workspace, thre_workspace, m_actual);
        AscendC::PipeBarrier<PIPE_V>();
        
        // AscendC::Add<ElementY, true>(
        //     thre_workspace,
        //     thre_workspace[4 * m_actual],
        //     thre_workspace,
        //     add_mask,
        //     CeilDiv(m_actual, elem_repeat_size),
        //     params);

        AscendC::Add<ElementY>(
            thre_workspace,
            thre_workspace[4 * m_actual],
            thre_workspace,
            m_actual);

        AscendC::PipeBarrier<PIPE_V>();

        // AscendC::Add<ElementY, true>(
        //     dstTensor,
        //     thre_workspace[m_actual],
        //     thre_workspace,
        //     add_mask,
        //     CeilDiv(m_actual, elem_repeat_size),
        //     params);

        AscendC::Add<ElementY>(
            thre_workspace,
            thre_workspace[m_actual],
            thre_workspace,
            m_actual);
        
        AscendC::PipeBarrier<PIPE_V>();


        AscendC::SetMaskCount();
        AscendC::SetVectorMask<ElementY, AscendC::MaskMode::COUNTER>(m_actual);
        AscendC::Muls<ElementY,false>(
            dstTensor,
            thre_workspace,
            e_max,
            AscendC::MASK_PLACEHOLDER,
            1,
            AscendC::UnaryRepeatParams{}
        );
        AscendC::SetMaskNorm();
        AscendC::ResetMask();
            
    }
};

/*
将threshold的计算矩阵化
*/

template <
    class ElementA
>
struct TileThreCalc<Arch::AtlasA2,
                helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
                Gemm::GemmType<ElementA, layout::RowMajor>,
                Gemm::GemmType<float, layout::RowMajor>,
                Gemm::GemmType<float, layout::RowMajor>,
                void>
{
    using ElementX = float;
    using ElementY = float;

    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementX, ElementY>::ElementAccumulator;

    using FT_THRESHOLD_ALGORITHM = helper::FT_THRESHOLD_ALGORITHM;

    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);
    static constexpr uint32_t OUT_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);

    // Mehtods

    CATLASS_DEVICE
    TileThreCalc() {};

    // 
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementY> srcMeanTensor,
        AscendC::LocalTensor<ElementY> srcStdTensor,
        AscendC::LocalTensor<ElementY> BMeanAbsTensor,
        AscendC::LocalTensor<ElementY> BMeanSquareTensor,
        AscendC::LocalTensor<ElementY> BVarTensor,
        AscendC::LocalTensor<ElementY> thre_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        ElementY n_ratio_factor, ElementY n_sqrt_ratio_factor,
        ElementY n_square_ratio_factor, 
        ElementY e_max
    )
    {

        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);

        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementX);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = elem_repeat_size;

        uint64_t add_mask = elem_repeat_size;
        uint64_t thre_mask = elem_repeat_size;
        uint64_t abs_mask = elem_repeat_size;

        uint32_t repeat_num = n_actual / elem_repeat_size;
        uint32_t remain = n_actual % elem_repeat_size;


        AscendC::BinaryRepeatParams thre_params;
        thre_params.dstBlkStride = 1;
        thre_params.src0BlkStride = 1;
        thre_params.src1BlkStride = 1;
        thre_params.dstRepStride = RoundUp(n_round, elem_repeat_size) / ELE_NUM_PER_C0;
        thre_params.src0RepStride = RoundUp(n_round, elem_repeat_size) / ELE_NUM_PER_C0;
        thre_params.src1RepStride = RoundUp(n_round, elem_repeat_size) / ELE_NUM_PER_C0;

        AscendC::UnaryRepeatParams final_params;

        final_params.dstBlkStride = 1;
        final_params.srcBlkStride = 1;
        final_params.dstRepStride = RoundUp(n_round, elem_repeat_size) / ELE_NUM_PER_C0;
        final_params.srcRepStride = RoundUp(n_round, elem_repeat_size) / ELE_NUM_PER_C0;


        AscendC::Duplicate<ElementY>(
            thre_workspace,
            (ElementY)0.0,
            int32_t(m_actual) * int32_t(n_round) * 5
        );

        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<AscendC::HardEvent::V_S>((event_t)(0));
        AscendC::WaitFlag<AscendC::HardEvent::V_S>((event_t)(0));

        /*
        template <typename T>
        __aicore__ inline void Abs(
            const LocalTensor<T>& dstLocal, 
            const LocalTensor<T>& srcLocal, const int32_t& calCount)
        */
        /*
        AscendC::MulAddDst<ElementAccumulator, ElementA, true>(
                vmad_workspace,
                srcTensor_m[offset], // 这里是起始地址，每次外层迭代都是从第一行的相应列偏移开始的
                srcTensor_v[offset],
                remain_mask,
                m_actual,
                vmad_params);
        */

        uint32_t term_2_base = 2 * m_actual * n_round;

        for(uint32_t i=0; i < repeat_num; i++){
            uint32_t offset = i * elem_repeat_size;
            AscendC::MulAddDst<ElementY, ElementX, true>(
                thre_workspace[term_2_base + offset],
                srcStdTensor[offset],
                srcStdTensor[offset],
                thre_mask,
                m_actual,
                thre_params
            );

            /*
            AscendC::MulAddDst<ElementY, ElementX>(
                thre_workspace[2 * m_actual],
                srcStdTensor,
                srcStdTensor,
                m_actual);
            */

            // AscendC::Abs(srcMeanTensor, srcMeanTensor, int32_t(m_actual));
            AscendC::Abs<ElementY, true>(
                srcMeanTensor[offset], 
                srcMeanTensor[offset], 
                abs_mask,
                m_actual,
                final_params
            );
        }

        AscendC::PipeBarrier<PIPE_V>();

        if(remain > 0){
            uint32_t offset = repeat_num * elem_repeat_size;
            if (offset + remain > n_actual)
            {
                remain = n_actual - offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;

            AscendC::MulAddDst<ElementY, ElementX, true>(
                thre_workspace[term_2_base + offset],
                srcStdTensor[offset],
                srcStdTensor[offset],
                remain_mask,
                m_actual,
                thre_params
            );

            // AscendC::Abs(srcMeanTensor[], srcMeanTensor, int32_t(m_actual));
            AscendC::Abs<ElementY, true>(
                srcMeanTensor[offset], 
                srcMeanTensor[offset], 
                remain_mask,
                m_actual,
                final_params
            );

            AscendC::PipeBarrier<PIPE_V>();
        }
 
        /*
        template <typename T, typename U>
        __aicore__ inline void MulAddDst(const LocalTensor<T>& dstLocal, 
            const LocalTensor<U>& src0Local, 
            const LocalTensor<U>& src1Local, 
            const int32_t& calCount)
        */
        // AscendC::MulAddDst<ElementY, ElementX>(
        //     thre_workspace[3 * m_actual],
        //     srcMeanTensor,
        //     srcMeanTensor,
        //     m_actual);

        uint32_t term_3_base = 3 * m_actual * n_round;

        for(uint32_t i=0; i < repeat_num; i++){
            uint32_t offset = i * elem_repeat_size;
            AscendC::MulAddDst<ElementY, ElementX, true>(
                thre_workspace[term_3_base + offset],
                srcMeanTensor[offset],
                srcMeanTensor[offset],
                thre_mask,
                m_actual,
                thre_params
            );

            /*
            AscendC::MulAddDst<ElementY, ElementX>(
                thre_workspace[3 * m_actual],
                srcMeanTensor,
                srcMeanTensor,
                m_actual);
            */
        }

        AscendC::PipeBarrier<PIPE_V>();

        if(remain > 0){
            uint32_t offset = repeat_num * elem_repeat_size;
            if (offset + remain > n_actual)
            {
                remain = n_actual - offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;

            AscendC::MulAddDst<ElementY, ElementX, true>(
                thre_workspace[term_3_base + offset],
                srcMeanTensor[offset],
                srcMeanTensor[offset],
                remain_mask,
                m_actual,
                thre_params
            );

            AscendC::PipeBarrier<PIPE_V>();
        }  

        AscendC::SetFlag<AscendC::HardEvent::V_S>((event_t)(0));
        

        uint32_t term_4_base = 4 * m_actual * n_round;
        uint32_t term_1_base = 1 * m_actual * n_round;

        // AscendC::SetMaskCount();
        // AscendC::SetVectorMask<ElementY, AscendC::MaskMode::COUNTER>(n_round);

        for(uint32_t j = 0; j < m_actual; j++){
            uint32_t row_offset = j * n_round;

            AscendC::WaitFlag<AscendC::HardEvent::V_S>((event_t)(0));
            float B_slice_meanabs = (float)BMeanAbsTensor.GetValue(j);
            float B_slice_meansquare = (float)BMeanSquareTensor.GetValue(j);
            float B_slice_var = (float)BVarTensor.GetValue(j);

            float B_slice_var_square = B_slice_var * B_slice_var;

            AscendC::SetFlag<AscendC::HardEvent::S_V>((event_t)(0));
            AscendC::WaitFlag<AscendC::HardEvent::S_V>((event_t)(0));

            // AscendC::Muls(dstLocal, srcLocal, scalar, 512);
            AscendC::Muls(
                thre_workspace[term_4_base + row_offset],
                srcMeanTensor[row_offset],
                n_ratio_factor*B_slice_meanabs,
                n_round
            );

            /*
            AscendC::Muls<ElementY,false>(
                thre_workspace[4 * m_actual],
                srcMeanTensor,
                n_ratio_factor*B_slice_meanabs,
                AscendC::MASK_PLACEHOLDER,
                1,
                AscendC::UnaryRepeatParams{}
            );
            */

            AscendC::Muls(
                thre_workspace[term_3_base + row_offset],
                thre_workspace[term_3_base + row_offset],
                16.0f * n_ratio_factor * B_slice_var_square,
                n_round
            );

            /*
            AscendC::Muls<ElementY,false>(
                thre_workspace[3 * m_actual],
                thre_workspace[3 * m_actual],
                16.0f * n_ratio_factor * B_slice_var_square,
                AscendC::MASK_PLACEHOLDER,
                1,
                AscendC::UnaryRepeatParams{}
            );
            */

            AscendC::Muls(
                thre_workspace[term_2_base + row_offset],
                thre_workspace[term_2_base + row_offset],
                16.0f * n_square_ratio_factor * B_slice_meansquare,
                n_round
            );

            /*
            AscendC::Muls<ElementY,false>(
                thre_workspace[m_actual * 2],
                thre_workspace[m_actual * 2],
                16.0f * n_square_ratio_factor * B_slice_meansquare,
                AscendC::MASK_PLACEHOLDER,
                1,
                AscendC::UnaryRepeatParams{}
            );
            */
            
            AscendC::Muls(
                thre_workspace[term_1_base + row_offset],
                srcStdTensor[row_offset],
                n_sqrt_ratio_factor * B_slice_var,
                n_round
            );

            /*
            AscendC::Muls<ElementY,false>(
                thre_workspace[m_actual],
                srcStdTensor,
                n_sqrt_ratio_factor * B_slice_var,
                AscendC::MASK_PLACEHOLDER,
                1,
                AscendC::UnaryRepeatParams{}
            );
            */

            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_S>((event_t)(0));
        }

        // AscendC::SetMaskNorm();
        // AscendC::ResetMask();
        
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WaitFlag<AscendC::HardEvent::V_S>((event_t)(0));

        AscendC::SetFlag<AscendC::HardEvent::S_V>((event_t)(0));
        AscendC::WaitFlag<AscendC::HardEvent::S_V>((event_t)(0));

        for(uint32_t i=0; i < repeat_num; i++){
            uint32_t offset = i * elem_repeat_size;
            AscendC::Add<ElementY, true>(
                thre_workspace[offset],
                thre_workspace[term_3_base + offset],
                thre_workspace[term_2_base + offset], 
                thre_mask,
                m_actual,
                thre_params
            );

            /*
            AscendC::Add<ElementY>(
                thre_workspace,
                thre_workspace[3 * m_actual],
                thre_workspace[2 * m_actual], 
                m_actual);
            */
        }

        AscendC::PipeBarrier<PIPE_V>();

        if(remain > 0){
            uint32_t offset = repeat_num * elem_repeat_size;
            if (offset + remain > n_actual)
            {
                remain = n_actual - offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;

            AscendC::Add<ElementY, true>(
                thre_workspace[offset],
                thre_workspace[term_3_base + offset],
                thre_workspace[term_2_base + offset],
                remain_mask,
                m_actual,
                thre_params
            );

            AscendC::PipeBarrier<PIPE_V>();
        }

        for(uint32_t i=0; i < repeat_num; i++){
            uint32_t offset = i * elem_repeat_size;
            
            // AscendC::Abs(thre_workspace, thre_workspace, m_actual);
            AscendC::Abs<ElementY, true>(
                thre_workspace[offset], 
                thre_workspace[offset], 
                abs_mask,
                m_actual,
                final_params
            );
        }

        AscendC::PipeBarrier<PIPE_V>();

        if(remain > 0){
            uint32_t offset = repeat_num * elem_repeat_size;
            if (offset + remain > n_actual)
            {
                remain = n_actual - offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;

            // AscendC::Abs(srcMeanTensor[], srcMeanTensor, int32_t(m_actual));
            AscendC::Abs<ElementY, true>(
                thre_workspace[offset], 
                thre_workspace[offset], 
                remain_mask,
                m_actual,
                final_params
            );

            AscendC::PipeBarrier<PIPE_V>();
        }

        for(uint32_t i=0; i < repeat_num; i++){
            uint32_t offset = i * elem_repeat_size;
            
            // AscendC::Sqrt(thre_workspace, thre_workspace, m_actual);
            AscendC::Sqrt<ElementY, true>(
                thre_workspace[offset], 
                thre_workspace[offset], 
                abs_mask,
                m_actual,
                final_params
            );
        }

        AscendC::PipeBarrier<PIPE_V>();

        if(remain > 0){
            uint32_t offset = repeat_num * elem_repeat_size;
            if (offset + remain > n_actual)
            {
                remain = n_actual - offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;

            // AscendC::Sqrt(thre_workspace, thre_workspace, m_actual);
            
            AscendC::Sqrt<ElementY, true>(
                thre_workspace[offset], 
                thre_workspace[offset], 
                remain_mask,
                m_actual,
                final_params
            );

            AscendC::PipeBarrier<PIPE_V>();
        }

        for(uint32_t i=0; i < repeat_num; i++){
            uint32_t offset = i * elem_repeat_size;
            AscendC::Add<ElementY, true>(
                thre_workspace[offset],
                thre_workspace[term_4_base + offset],
                thre_workspace[offset], 
                thre_mask,
                m_actual,
                thre_params
            );

            /*
            AscendC::Add<ElementY>(
                thre_workspace,
                thre_workspace[4 * m_actual],
                thre_workspace, 
                m_actual);
            */
        }

        AscendC::PipeBarrier<PIPE_V>();

        if(remain > 0){
            uint32_t offset = repeat_num * elem_repeat_size;
            if (offset + remain > n_actual)
            {
                remain = n_actual - offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;

            AscendC::Add<ElementY, true>(
                thre_workspace[offset],
                thre_workspace[term_4_base + offset],
                thre_workspace[offset], 
                remain_mask,
                m_actual,
                thre_params
            );

            AscendC::PipeBarrier<PIPE_V>();
        }

        for(uint32_t i=0; i < repeat_num; i++){
            uint32_t offset = i * elem_repeat_size;
            AscendC::Add<ElementY, true>(
                thre_workspace[offset],
                thre_workspace[term_1_base + offset],
                thre_workspace[offset], 
                thre_mask,
                m_actual,
                thre_params
            );

            /*
            AscendC::Add<ElementY>(
                thre_workspace,
                thre_workspace[m_actual],
                thre_workspace, 
                m_actual);
            */
        }

        AscendC::PipeBarrier<PIPE_V>();

        if(remain > 0){
            uint32_t offset = repeat_num * elem_repeat_size;
            if (offset + remain > n_actual)
            {
                remain = n_actual - offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;

            AscendC::Add<ElementY, true>(
                thre_workspace[offset],
                thre_workspace[term_1_base + offset],
                thre_workspace[offset], 
                remain_mask,
                m_actual,
                thre_params
            );

            AscendC::PipeBarrier<PIPE_V>();
        }


        // AscendC::SetMaskCount();
        // AscendC::SetVectorMask<ElementY, AscendC::MaskMode::COUNTER>(m_actual);

        for(uint32_t i=0; i < repeat_num; i++){
            uint32_t offset = i * elem_repeat_size;
            AscendC::Muls<ElementY, true>(
                dstTensor[offset],
                thre_workspace[offset], 
                e_max,
                abs_mask,
                m_actual,
                final_params
            );

            /*
            AscendC::Muls<ElementY,false>(
                dstTensor,
                thre_workspace,
                e_max,
                AscendC::MASK_PLACEHOLDER,
                1,
                AscendC::UnaryRepeatParams{}
            );
            */
        }

        AscendC::PipeBarrier<PIPE_V>();

        if(remain > 0){
            uint32_t offset = repeat_num * elem_repeat_size;
            if (offset + remain > n_actual)
            {
                remain = n_actual - offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;

            AscendC::Muls<ElementY, true>(
                dstTensor[offset],
                thre_workspace[offset], 
                e_max,
                remain_mask,
                m_actual,
                final_params
            );
        }

        
        // AscendC::SetMaskNorm();
        // AscendC::ResetMask();
        
        AscendC::PipeBarrier<PIPE_V>();
    }
};

template <
    class ElementA
>
struct TileThreCalcChunk<Arch::AtlasA2,
        helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
        Gemm::GemmType<ElementA, layout::RowMajor>,
        Gemm::GemmType<float, layout::VectorLayout>,
        Gemm::GemmType<float, layout::VectorLayout>,
        void>
{
    using ElementX = float;
    using ElementY = float;

    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementX, ElementY>::ElementAccumulator;

    using FT_THRESHOLD_ALGORITHM = helper::FT_THRESHOLD_ALGORITHM;

    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST;
    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::VectorLayout;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);

    // Mehtods

    CATLASS_DEVICE
    TileThreCalcChunk() {};

    // 
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementX> srcMeanTensor,
        AscendC::LocalTensor<ElementY> srcStdTensor,
        AscendC::LocalTensor<ElementY> BMeanAbsTensor,
        AscendC::LocalTensor<ElementY> BMeanSquareTensor,
        AscendC::LocalTensor<ElementY> BVarTensor,
        AscendC::LocalTensor<ElementY> thre_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrcA, 
        LayoutSrc const &layoutSrcB,
        ElementY n_ratio_factor, ElementY n_sqrt_ratio_factor,
        ElementY n_square_ratio_factor,
        ElementY n_ratio_factor_tail, 
        ElementY n_sqrt_ratio_factor_tail, 
        ElementY n_square_ratio_factor_tail,
        ElementY e_max)
    {
        uint32_t n_actual = layoutSrcA.shape(0);
        uint32_t m_actual = layoutSrcB.shape(0);

        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);

        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementX);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = elem_repeat_size;

        uint64_t add_mask = elem_repeat_size;
        uint32_t repeattimes = m_actual;

        AscendC::BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;
        params.dstRepStride = 8;
        params.src0RepStride = 8;
        params.src1RepStride = 8;

        AscendC::Duplicate<ElementY>(
            thre_workspace[int32_t(n_round) * 5],
            (ElementY)0.0,
            int32_t(n_round) * 2);
        
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::MulAddDst<ElementY, ElementX>(
            thre_workspace[int32_t(n_round) * 5],
            srcStdTensor,
            srcStdTensor,
            n_actual);

        AscendC::Abs(srcMeanTensor, srcMeanTensor, int32_t(n_actual));
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::MulAddDst<ElementY, ElementX>(
            thre_workspace[int32_t(n_round) * 6],
            srcMeanTensor,
            srcMeanTensor,
            n_actual);
            
        AscendC::PipeBarrier<PIPE_V>();
        
        for(uint32_t j=0; j < m_actual; j++){

            uint32_t row_offset = j * n_round;

            AscendC::Duplicate<ElementY>(
                thre_workspace,
                (ElementY)0.0,
                int32_t(n_round) * 5
            );

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::SetFlag<AscendC::HardEvent::V_S>((event_t)(0));
            AscendC::WaitFlag<AscendC::HardEvent::V_S>((event_t)(0));

            float B_slice_meanabs = (float)BMeanAbsTensor.GetValue(j);
            float B_slice_meansquare = (float)BMeanSquareTensor.GetValue(j);
            float B_slice_var = (float)BVarTensor.GetValue(j);

            float B_slice_var_square = B_slice_var * B_slice_var;

            ElementY n_ratio_factor_used = n_ratio_factor;
            ElementY n_sqrt_ratio_factor_used = n_sqrt_ratio_factor;
            ElementY n_square_ratio_factor_used = n_square_ratio_factor;

            if(j == (m_actual - 1)){
                n_ratio_factor_used = n_ratio_factor_tail;
                n_sqrt_ratio_factor_used = n_sqrt_ratio_factor_tail;
                n_square_ratio_factor_used = n_square_ratio_factor;
            }

            AscendC::SetFlag<AscendC::HardEvent::S_V>((event_t)(0));
            AscendC::WaitFlag<AscendC::HardEvent::S_V>((event_t)(0));
            /*
            template <typename T>
            __aicore__ inline void Abs(
                const LocalTensor<T>& dstLocal, 
                const LocalTensor<T>& srcLocal, const int32_t& calCount)
            */

            // AscendC::MulAddDst<ElementY, ElementX>(
            //     thre_workspace[2 * n_round],
            //     srcStdTensor,
            //     srcStdTensor,
            //     n_round);

            /*
            template <typename T, typename U>
            __aicore__ inline void MulAddDst(const LocalTensor<T>& dstLocal, 
                const LocalTensor<U>& src0Local, 
                const LocalTensor<U>& src1Local, 
                const int32_t& calCount)
            */

            // AscendC::MulAddDst<ElementY, ElementX>(
            //     thre_workspace[3 * n_round],
            //     srcMeanTensor,
            //     srcMeanTensor,
            //     n_round);

            AscendC::SetMaskCount();
            AscendC::SetVectorMask<ElementY, AscendC::MaskMode::COUNTER>(n_round);
            AscendC::Muls<ElementY,false>(
                thre_workspace[4 * n_round],
                srcMeanTensor,
                n_ratio_factor_used * B_slice_meanabs,
                AscendC::MASK_PLACEHOLDER,
                1,
                AscendC::UnaryRepeatParams{}
            );

            AscendC::Muls<ElementY,false>(
                thre_workspace[3 * n_round],
                thre_workspace[int32_t(n_round) * 6],
                16.0f * n_ratio_factor_used * B_slice_var_square,
                AscendC::MASK_PLACEHOLDER,
                1,
                AscendC::UnaryRepeatParams{}
            );

            AscendC::Muls<ElementY,false>(
                thre_workspace[n_round * 2],
                thre_workspace[int32_t(n_round) * 5],
                16.0f * n_square_ratio_factor_used * B_slice_meansquare,
                AscendC::MASK_PLACEHOLDER,
                1,
                AscendC::UnaryRepeatParams{}
            );

            AscendC::Muls<ElementY,false>(
                thre_workspace[n_round],
                srcStdTensor,
                n_sqrt_ratio_factor_used * B_slice_var,
                AscendC::MASK_PLACEHOLDER,
                1,
                AscendC::UnaryRepeatParams{}
            );
            AscendC::SetMaskNorm();
            AscendC::ResetMask();

            AscendC::PipeBarrier<PIPE_V>();

            uint32_t add_mask_tmp = (n_round < elem_repeat_size) ? n_round : elem_repeat_size;

            // AscendC::Add<ElementY, true>(
            //     thre_workspace,
            //     thre_workspace[3 * n_round],
            //     thre_workspace[2 * n_round],
            //     add_mask,
            //     CeilDiv(n_round, elem_repeat_size),
            //     params);
            
            /*
            template <typename T>
            __aicore__ inline void 
            Add(const LocalTensor<T>& dstLocal, 
                const LocalTensor<T>& src0Local, 
                const LocalTensor<T>& src1Local, 
                const int32_t& calCount)
            */
            AscendC::Add<ElementY>(
                thre_workspace,
                thre_workspace[3 * n_round],
                thre_workspace[2 * n_round], 
                n_round);

             // AscendC::Abs(thre_workspace, thre_workspace, n_round);
        
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Abs(thre_workspace, thre_workspace, n_round);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sqrt(thre_workspace, thre_workspace, n_round);
            AscendC::PipeBarrier<PIPE_V>();
        
            // AscendC::Add<ElementY, true>(
            //     thre_workspace,
            //     thre_workspace[4 * n_round],
            //     thre_workspace,
            //     add_mask,
            //     CeilDiv(n_round, elem_repeat_size),
            //     params);
        
            AscendC::Add<ElementY>(
                thre_workspace,
                thre_workspace[4 * n_round],
                thre_workspace, 
                n_round);

            AscendC::PipeBarrier<PIPE_V>();

            // AscendC::Add<ElementY, true>(
            //     dstTensor,
            //     thre_workspace[n_round],
            //     thre_workspace,
            //     add_mask,
            //     CeilDiv(n_round, elem_repeat_size),
            //     params);

            AscendC::Add<ElementY>(
                thre_workspace,
                thre_workspace[n_round],
                thre_workspace, 
                n_round);
        
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::SetMaskCount();
            AscendC::SetVectorMask<ElementY, AscendC::MaskMode::COUNTER>(n_round);
            AscendC::Muls<ElementY,false>(
                dstTensor[row_offset],
                thre_workspace,
                e_max,
                AscendC::MASK_PLACEHOLDER,
                1,
                AscendC::UnaryRepeatParams{}
            );
            AscendC::SetMaskNorm();
            AscendC::ResetMask();
        
            AscendC::PipeBarrier<PIPE_V>();
        }       
    }
};

}

#endif