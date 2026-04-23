#ifndef CATLASS_GEMV_TILE_TILE_STD_ESTIMATE_HPP
#define CATLASS_GEMV_TILE_TILE_STD_ESTIMATE_HPP

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
    class XType,
    class YType,
    class BiasType = void
>
struct TileStdEst
{
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported TileStdEst, can not find the specialization.");
};

template <
    /// Tag indicating architecture
    class ArchTag,
    class XType,
    class YType,
    class BiasType = void
>
struct TileStdEstRobust
{
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported TileStdEstRobust, can not find the specialization.");
};


template <
    /// Tag indicating architecture
    class ArchTag,
    class XType,
    class YType,
    class BiasType = void
>
struct TileStdEstSimplified
{
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported TileStdEstSimplified, can not find the specialization.");
};

template <
    class ElementY
>
struct TileStdEst<Arch::AtlasA2,
                Gemm::GemmType<ElementY, layout::VectorLayout>,
                Gemm::GemmType<ElementY, layout::VectorLayout>,
                void>
{
    using ElementX = ElementY;
    // using ElementY = half;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementX, ElementY>::ElementAccumulator;

    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementX);
    static constexpr uint32_t OUT_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);
    // Mehtods

    CATLASS_DEVICE
    TileStdEst() {};

    // 
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementY> srcMeanTensor,
        AscendC::LocalTensor<ElementY> srcMaxTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        ElementY std_scale_factor
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t m_round = layoutDst.shape(0);

        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementY);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = temp_repeat_size;
        uint32_t repeattimes = CeilDiv(m_actual, temp_repeat_size);

        /*
        template <typename T>
        __aicore__ inline void Sub(const LocalTensor<T>& dstLocal, 
            const LocalTensor<T>& src0Local, 
            const LocalTensor<T>& src1Local, 
            const int32_t& calCount)
        */

        AscendC::Sub(dstTensor,srcMaxTensor,srcMeanTensor,m_actual);
        AscendC::PipeBarrier<PIPE_V>();

        

        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Muls(const LocalTensor<T>& dstLocal, 
        const LocalTensor<T>& srcLocal, 
        const T& scalarValue, 
        const int32_t& calCount)
        */

        AscendC::SetMaskCount();
        AscendC::SetVectorMask<ElementY, AscendC::MaskMode::COUNTER>(m_actual);
        AscendC::Muls<ElementY,false>(
            dstTensor,
            dstTensor,
            std_scale_factor,
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
    class ElementX,
    class ElementY
>
struct TileStdEst<Arch::AtlasA2,
                Gemm::GemmType<ElementX, layout::VectorLayout>,
                Gemm::GemmType<ElementY, layout::VectorLayout>,
                void>
{
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementX, ElementY>::ElementAccumulator;

    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementX);
    static constexpr uint32_t OUT_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);
    // Mehtods

    CATLASS_DEVICE
    TileStdEst() {};

    // 
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementX> srcMeanTensor,
        AscendC::LocalTensor<ElementX> srcMaxTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        ElementY std_scale_factor
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t m_round = layoutDst.shape(0);

        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementX);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = temp_repeat_size;
        uint32_t repeattimes = CeilDiv(m_actual, temp_repeat_size);

        /*
        template <typename T>
        __aicore__ inline void Sub(const LocalTensor<T>& dstLocal, 
            const LocalTensor<T>& src0Local, 
            const LocalTensor<T>& src1Local, 
            const int32_t& calCount)
        */

        AscendC::Sub(srcMaxTensor,srcMaxTensor,srcMeanTensor,m_actual);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::UnaryRepeatParams castparams;
        castparams.dstBlkStride = 1;
        castparams.srcBlkStride = 1;
        castparams.dstRepStride = 8;
        castparams.srcRepStride = 4;

        /*
        template <typename T1, typename T2>
        __aicore__ inline void Cast(const LocalTensor<T1>& dstLocal, 
            const LocalTensor<T2>& srcLocal, 
            const RoundMode& round_mode, const uint32_t calCount)

        (uint64_t)mask, repeattimes, castparams
        */
        AscendC::Cast<ElementY, ElementX>(
            dstTensor,
            srcMaxTensor,
            AscendC::RoundMode::CAST_NONE, m_actual);
        AscendC::PipeBarrier<PIPE_V>();

        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Muls(const LocalTensor<T>& dstLocal, 
        const LocalTensor<T>& srcLocal, 
        const T& scalarValue, 
        const int32_t& calCount)
        */

        AscendC::SetMaskCount();
        AscendC::SetVectorMask<ElementY, AscendC::MaskMode::COUNTER>(m_actual);
        AscendC::Muls<ElementY,false>(
            dstTensor,
            dstTensor,
            std_scale_factor,
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
    class ElementY
>
struct TileStdEstRobust<Arch::AtlasA2,
                Gemm::GemmType<ElementY, layout::VectorLayout>,
                Gemm::GemmType<ElementY, layout::VectorLayout>,
                void>
{
    using ElementX = ElementY;
    // using ElementY = half;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementX, ElementY>::ElementAccumulator;

    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementX);
    static constexpr uint32_t OUT_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);
    // Mehtods

    CATLASS_DEVICE
    TileStdEstRobust() {};

    // 
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementY> srcMeanTensor,
        AscendC::LocalTensor<ElementY> srcMaxTensor,
        AscendC::LocalTensor<ElementY> srcMinTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t m_round = layoutDst.shape(0);

        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementY);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = temp_repeat_size;
        uint32_t repeattimes = CeilDiv(m_actual, temp_repeat_size);

        /*
        template <typename T>
        __aicore__ inline void Sub(const LocalTensor<T>& dstLocal, 
            const LocalTensor<T>& src0Local, 
            const LocalTensor<T>& src1Local, 
            const int32_t& calCount)
        */

        AscendC::Sub(srcMaxTensor, srcMaxTensor, srcMeanTensor, m_actual);
        AscendC::Sub(srcMinTensor, srcMeanTensor, srcMinTensor, m_actual);
        AscendC::PipeBarrier<PIPE_V>();
        /*
        template <typename T>
        __aicore__ inline void Mul(
            const LocalTensor<T>& dstLocal, 
            const LocalTensor<T>& src0Local, 
            const LocalTensor<T>& src1Local, 
            const int32_t& calCount)
        */
        AscendC::Mul(srcMaxTensor, srcMaxTensor, srcMinTensor, m_actual);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Abs(srcMaxTensor, srcMaxTensor, m_actual);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sqrt(dstTensor, srcMaxTensor, m_actual);
        AscendC::PipeBarrier<PIPE_V>();
    }
};

template <
>
struct TileStdEstRobust<Arch::AtlasA2,
                Gemm::GemmType<float, layout::VectorLayout>,
                Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementX = float;
    using ElementY = float;
    // using ElementY = half;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementX, ElementY>::ElementAccumulator;

    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementX);
    static constexpr uint32_t OUT_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);
    // Mehtods

    CATLASS_DEVICE
    TileStdEstRobust() {};

    // 
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementY> srcMeanTensor,
        AscendC::LocalTensor<ElementY> srcMaxTensor,
        AscendC::LocalTensor<ElementY> srcMinTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t m_round = layoutDst.shape(0);

        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementY);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = temp_repeat_size;
        uint32_t repeattimes = CeilDiv(m_actual, temp_repeat_size);

        /*
        template <typename T>
        __aicore__ inline void Sub(const LocalTensor<T>& dstLocal, 
            const LocalTensor<T>& src0Local, 
            const LocalTensor<T>& src1Local, 
            const int32_t& calCount)
        */

        AscendC::Sub(srcMaxTensor, srcMaxTensor, srcMeanTensor, m_actual);
        AscendC::Sub(srcMinTensor, srcMeanTensor, srcMinTensor, m_actual);
        AscendC::PipeBarrier<PIPE_V>();
        /*
        template <typename T>
        __aicore__ inline void Mul(
            const LocalTensor<T>& dstLocal, 
            const LocalTensor<T>& src0Local, 
            const LocalTensor<T>& src1Local, 
            const int32_t& calCount)
        */
        AscendC::Mul(srcMaxTensor, srcMaxTensor, srcMinTensor, m_actual);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Abs(srcMaxTensor, srcMaxTensor, m_actual);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sqrt(dstTensor, srcMaxTensor, m_actual);
        AscendC::PipeBarrier<PIPE_V>();
    }
};


template <
    class ElementX,
    class ElementY
>
struct TileStdEstRobust<Arch::AtlasA2,
                Gemm::GemmType<ElementX, layout::VectorLayout>,
                Gemm::GemmType<ElementY, layout::VectorLayout>,
                void>
{
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementX, ElementX>::ElementAccumulator;

    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementX);
    static constexpr uint32_t OUT_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);
    // Mehtods

    CATLASS_DEVICE
    TileStdEstRobust() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementY> srcMeanTensor,
        AscendC::LocalTensor<ElementY> srcMaxTensor,
        AscendC::LocalTensor<ElementY> srcMinTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t m_round = layoutDst.shape(0);

        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementY);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = temp_repeat_size;
        uint32_t repeattimes = CeilDiv(m_actual, temp_repeat_size);

        /*
        template <typename T>
        __aicore__ inline void Sub(const LocalTensor<T>& dstLocal, 
            const LocalTensor<T>& src0Local, 
            const LocalTensor<T>& src1Local, 
            const int32_t& calCount)
        */

        AscendC::Sub(srcMaxTensor, srcMaxTensor,srcMeanTensor,m_actual);
        AscendC::Sub(srcMinTensor, srcMeanTensor, srcMinTensor, m_actual);
        AscendC::PipeBarrier<PIPE_V>();
        /*
        template <typename T>
        __aicore__ inline void Mul(
            const LocalTensor<T>& dstLocal, 
            const LocalTensor<T>& src0Local, 
            const LocalTensor<T>& src1Local, 
            const int32_t& calCount)
        */
        AscendC::Mul(srcMaxTensor, srcMaxTensor, srcMinTensor, m_actual);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Abs(srcMaxTensor, srcMaxTensor, m_actual);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sqrt(srcMaxTensor, srcMaxTensor, m_actual);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::UnaryRepeatParams castparams;
        castparams.dstBlkStride = 1;
        castparams.srcBlkStride = 1;
        castparams.dstRepStride = 8;
        castparams.srcRepStride = 4;

        /*
        template <typename T1, typename T2>
        __aicore__ inline void Cast(const LocalTensor<T1>& dstLocal, 
            const LocalTensor<T2>& srcLocal, 
            const RoundMode& round_mode, const uint32_t calCount)

        (uint64_t)mask, repeattimes, castparams
        */
        AscendC::Cast<ElementY, ElementX>(
            dstTensor,
            srcMaxTensor,
            AscendC::RoundMode::CAST_NONE, m_actual);
        AscendC::PipeBarrier<PIPE_V>();
    }
};



template <
    class ElementY
>
struct TileStdEstSimplified<Arch::AtlasA2,
                Gemm::GemmType<ElementY, layout::VectorLayout>,
                Gemm::GemmType<ElementY, layout::VectorLayout>,
                void>
{
    using ElementX = ElementY;
    // using ElementY = half;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementX, ElementY>::ElementAccumulator;

    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementX);
    static constexpr uint32_t OUT_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);
    // Mehtods

    CATLASS_DEVICE
    TileStdEstSimplified() {};

    // 
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementY> srcMaxTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        ElementY std_scale_factor
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t m_round = layoutDst.shape(0);

        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementY);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = temp_repeat_size;
        uint32_t repeattimes = CeilDiv(m_actual, temp_repeat_size);

        /*
        template <typename T>
        __aicore__ inline void Sub(const LocalTensor<T>& dstLocal, 
            const LocalTensor<T>& src0Local, 
            const LocalTensor<T>& src1Local, 
            const int32_t& calCount)
        */

        // AscendC::Sub(dstTensor,srcMaxTensor,srcMeanTensor,m_actual);
        // AscendC::PipeBarrier<PIPE_V>();

        

        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Muls(const LocalTensor<T>& dstLocal, 
        const LocalTensor<T>& srcLocal, 
        const T& scalarValue, 
        const int32_t& calCount)
        */

        AscendC::SetMaskCount();
        AscendC::SetVectorMask<ElementY, AscendC::MaskMode::COUNTER>(m_actual);
        AscendC::Muls<ElementY,false>(
            dstTensor,
            srcMaxTensor,
            std_scale_factor,
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
    class ElementX,
    class ElementY
>
struct TileStdEstSimplified<Arch::AtlasA2,
                Gemm::GemmType<ElementX, layout::VectorLayout>,
                Gemm::GemmType<ElementY, layout::VectorLayout>,
                void>
{
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementX, ElementY>::ElementAccumulator;

    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementX);
    static constexpr uint32_t OUT_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);
    // Mehtods

    CATLASS_DEVICE
    TileStdEstSimplified() {};

    // 
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementX> srcMaxTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        ElementY std_scale_factor
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t m_round = layoutDst.shape(0);

        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementX);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = temp_repeat_size;
        uint32_t repeattimes = CeilDiv(m_actual, temp_repeat_size);

        /*
        template <typename T>
        __aicore__ inline void Sub(const LocalTensor<T>& dstLocal, 
            const LocalTensor<T>& src0Local, 
            const LocalTensor<T>& src1Local, 
            const int32_t& calCount)
        */

        // AscendC::Sub(srcMaxTensor,srcMaxTensor,srcMeanTensor,m_actual);
        // AscendC::PipeBarrier<PIPE_V>();

        AscendC::UnaryRepeatParams castparams;
        castparams.dstBlkStride = 1;
        castparams.srcBlkStride = 1;
        castparams.dstRepStride = 8;
        castparams.srcRepStride = 4;

        /*
        template <typename T1, typename T2>
        __aicore__ inline void Cast(const LocalTensor<T1>& dstLocal, 
            const LocalTensor<T2>& srcLocal, 
            const RoundMode& round_mode, const uint32_t calCount)

        (uint64_t)mask, repeattimes, castparams
        */
        AscendC::Cast<ElementY, ElementX>(
            dstTensor,
            srcMaxTensor,
            AscendC::RoundMode::CAST_NONE, m_actual);
        AscendC::PipeBarrier<PIPE_V>();

        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Muls(const LocalTensor<T>& dstLocal, 
        const LocalTensor<T>& srcLocal, 
        const T& scalarValue, 
        const int32_t& calCount)
        */

        AscendC::SetMaskCount();
        AscendC::SetVectorMask<ElementY, AscendC::MaskMode::COUNTER>(m_actual);
        AscendC::Muls<ElementY,false>(
            dstTensor,
            dstTensor,
            std_scale_factor,
            AscendC::MASK_PLACEHOLDER,
            1,
            AscendC::UnaryRepeatParams{}
        );
        AscendC::SetMaskNorm();
        AscendC::ResetMask();

        AscendC::PipeBarrier<PIPE_V>();
    }
};


/*
    class ElementA,
    class ElementX,
    class ElementY
*/

template <>
struct TileStdEst<Arch::AtlasA2,
                Gemm::GemmType<float, layout::VectorLayout>,
                Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementX = float;
    using ElementY = float;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementX, ElementY>::ElementAccumulator;

    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementX);
    static constexpr uint32_t OUT_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);
    // Mehtods

    CATLASS_DEVICE
    TileStdEst() {};

    // 
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementX> srcMeanTensor,
        AscendC::LocalTensor<ElementX> srcMaxTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        ElementY std_scale_factor
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t m_round = layoutDst.shape(0);

        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementX);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = temp_repeat_size;
        uint32_t repeattimes = CeilDiv(m_actual, temp_repeat_size);

        /*
        template <typename T>
        __aicore__ inline void Sub(const LocalTensor<T>& dstLocal, 
            const LocalTensor<T>& src0Local, 
            const LocalTensor<T>& src1Local, 
            const int32_t& calCount)
        */

        AscendC::Sub(dstTensor,srcMaxTensor,srcMeanTensor,m_actual);
        AscendC::PipeBarrier<PIPE_V>();  

        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Muls(const LocalTensor<T>& dstLocal, 
        const LocalTensor<T>& srcLocal, 
        const T& scalarValue, 
        const int32_t& calCount)
        */

        AscendC::SetMaskCount();
        AscendC::SetVectorMask<ElementY, AscendC::MaskMode::COUNTER>(m_actual);
        AscendC::Muls<ElementY,false>(
            dstTensor,
            dstTensor,
            std_scale_factor,
            AscendC::MASK_PLACEHOLDER,
            1,
            AscendC::UnaryRepeatParams{}
        );
        AscendC::SetMaskNorm();
        AscendC::ResetMask();

        AscendC::PipeBarrier<PIPE_V>();
    }
};


template <>
struct TileStdEst<Arch::AtlasA2,
                Gemm::GemmType<half, layout::VectorLayout>,
                Gemm::GemmType<half, layout::VectorLayout>,
                void>
{
    using ElementX = half;
    using ElementY = half;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementX, ElementY>::ElementAccumulator;

    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementX);
    static constexpr uint32_t OUT_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);
    // Mehtods

    CATLASS_DEVICE
    TileStdEst() {};

    // 
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementX> srcMeanTensor,
        AscendC::LocalTensor<ElementX> srcMaxTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        ElementY std_scale_factor
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t m_round = layoutDst.shape(0);

        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementX);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = temp_repeat_size;
        uint32_t repeattimes = CeilDiv(m_actual, temp_repeat_size);

        /*
        template <typename T>
        __aicore__ inline void Sub(const LocalTensor<T>& dstLocal, 
            const LocalTensor<T>& src0Local, 
            const LocalTensor<T>& src1Local, 
            const int32_t& calCount)
        */

        AscendC::Sub(dstTensor,srcMaxTensor,srcMeanTensor,m_actual);
        AscendC::PipeBarrier<PIPE_V>();

        
        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Muls(const LocalTensor<T>& dstLocal, 
        const LocalTensor<T>& srcLocal, 
        const T& scalarValue, 
        const int32_t& calCount)
        */

        AscendC::SetMaskCount();
        AscendC::SetVectorMask<ElementY, AscendC::MaskMode::COUNTER>(m_actual);
        AscendC::Muls<ElementY,false>(
            dstTensor,
            dstTensor,
            std_scale_factor,
            AscendC::MASK_PLACEHOLDER,
            1,
            AscendC::UnaryRepeatParams{}
        );
        AscendC::SetMaskNorm();
        AscendC::ResetMask();

        AscendC::PipeBarrier<PIPE_V>();
    }
};

// class ElementX,
// class ElementY
template <>
struct TileStdEst<Arch::AtlasA2,
                Gemm::GemmType<half, layout::VectorLayout>,
                Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementX = half;
    using ElementY = float;

    using ElementAccumulator = float;
        
    // typename Gemm::helper::ElementAccumulatorSelector<ElementX, ElementY>::ElementAccumulator;

    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementX);
    static constexpr uint32_t OUT_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);
    // Mehtods

    CATLASS_DEVICE
    TileStdEst() {};

    

    // 
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementX> srcMeanTensor,
        AscendC::LocalTensor<ElementX> srcMaxTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        ElementY std_scale_factor
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t m_round = layoutDst.shape(0);

        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementX);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = temp_repeat_size;
        uint32_t repeattimes = CeilDiv(m_actual, temp_repeat_size);

        /*
        template <typename T>
        __aicore__ inline void Sub(const LocalTensor<T>& dstLocal, 
            const LocalTensor<T>& src0Local, 
            const LocalTensor<T>& src1Local, 
            const int32_t& calCount)
        */

        AscendC::Sub(srcMaxTensor,srcMaxTensor,srcMeanTensor,m_actual);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::UnaryRepeatParams castparams;
        castparams.dstBlkStride = 1;
        castparams.srcBlkStride = 1;
        castparams.dstRepStride = 8;
        castparams.srcRepStride = 4;

        /*
        template <typename T1, typename T2>
        __aicore__ inline void Cast(const LocalTensor<T1>& dstLocal, 
            const LocalTensor<T2>& srcLocal, 
            const RoundMode& round_mode, const uint32_t calCount)

        (uint64_t)mask, repeattimes, castparams
        */
        AscendC::Cast<ElementY, ElementX>(
            dstTensor,
            srcMaxTensor,
            AscendC::RoundMode::CAST_NONE, m_actual);
        AscendC::PipeBarrier<PIPE_V>();

        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Muls(const LocalTensor<T>& dstLocal, 
        const LocalTensor<T>& srcLocal, 
        const T& scalarValue, 
        const int32_t& calCount)
        */

        AscendC::SetMaskCount();
        AscendC::SetVectorMask<ElementY, AscendC::MaskMode::COUNTER>(m_actual);
        AscendC::Muls<ElementY,false>(
            dstTensor,
            dstTensor,
            std_scale_factor,
            AscendC::MASK_PLACEHOLDER,
            1,
            AscendC::UnaryRepeatParams{}
        );
        AscendC::SetMaskNorm();
        AscendC::ResetMask();

        AscendC::PipeBarrier<PIPE_V>();
    }
};


template <>
struct TileStdEstRobust<Arch::AtlasA2,
                Gemm::GemmType<half, layout::VectorLayout>,
                Gemm::GemmType<half, layout::VectorLayout>,
                void>
{
    using ElementX = half;
    using ElementY = half;
    // using ElementY = half;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementX, ElementY>::ElementAccumulator;

    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementX);
    static constexpr uint32_t OUT_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);
    // Mehtods

    CATLASS_DEVICE
    TileStdEstRobust() {};

    // 
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementY> srcMeanTensor,
        AscendC::LocalTensor<ElementY> srcMaxTensor,
        AscendC::LocalTensor<ElementY> srcMinTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t m_round = layoutDst.shape(0);

        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementY);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = temp_repeat_size;
        uint32_t repeattimes = CeilDiv(m_actual, temp_repeat_size);

        /*
        template <typename T>
        __aicore__ inline void Sub(const LocalTensor<T>& dstLocal, 
            const LocalTensor<T>& src0Local, 
            const LocalTensor<T>& src1Local, 
            const int32_t& calCount)
        */

        AscendC::Sub(srcMaxTensor, srcMaxTensor,srcMeanTensor,m_actual);
        AscendC::Sub(srcMinTensor, srcMeanTensor, srcMinTensor, m_actual);
        AscendC::PipeBarrier<PIPE_V>();
        /*
        template <typename T>
        __aicore__ inline void Mul(
            const LocalTensor<T>& dstLocal, 
            const LocalTensor<T>& src0Local, 
            const LocalTensor<T>& src1Local, 
            const int32_t& calCount)
        */
        AscendC::Mul(srcMaxTensor, srcMaxTensor, srcMinTensor, m_actual);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Abs(srcMaxTensor, srcMaxTensor, m_actual);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sqrt(dstTensor, srcMaxTensor, m_actual);
        AscendC::PipeBarrier<PIPE_V>();
    }
};

template <>
struct TileStdEstRobust<Arch::AtlasA2,
                Gemm::GemmType<half, layout::VectorLayout>,
                Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementX = half;
    using ElementY = float;
    // using ElementY = half;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementX, ElementX>::ElementAccumulator;

    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementX);
    static constexpr uint32_t OUT_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);
    // Mehtods

    CATLASS_DEVICE
    TileStdEstRobust() {};

    // 
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementY> srcMeanTensor,
        AscendC::LocalTensor<ElementY> srcMaxTensor,
        AscendC::LocalTensor<ElementY> srcMinTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t m_round = layoutDst.shape(0);

        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementY);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = temp_repeat_size;
        uint32_t repeattimes = CeilDiv(m_actual, temp_repeat_size);

        /*
        template <typename T>
        __aicore__ inline void Sub(const LocalTensor<T>& dstLocal, 
            const LocalTensor<T>& src0Local, 
            const LocalTensor<T>& src1Local, 
            const int32_t& calCount)
        */

        AscendC::Sub(srcMaxTensor, srcMaxTensor,srcMeanTensor,m_actual);
        AscendC::Sub(srcMinTensor, srcMeanTensor, srcMinTensor, m_actual);
        AscendC::PipeBarrier<PIPE_V>();
        /*
        template <typename T>
        __aicore__ inline void Mul(
            const LocalTensor<T>& dstLocal, 
            const LocalTensor<T>& src0Local, 
            const LocalTensor<T>& src1Local, 
            const int32_t& calCount)
        */
        AscendC::Mul(srcMaxTensor, srcMaxTensor, srcMinTensor, m_actual);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Abs(srcMaxTensor, srcMaxTensor, m_actual);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sqrt(dstTensor, srcMaxTensor, m_actual);
        AscendC::PipeBarrier<PIPE_V>();
    }
};



template <>
struct TileStdEstSimplified<Arch::AtlasA2,
                Gemm::GemmType<float, layout::VectorLayout>,
                Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementX = float;
    using ElementY = float;
    // using ElementY = half;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementX, ElementY>::ElementAccumulator;

    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementX);
    static constexpr uint32_t OUT_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);
    // Mehtods

    CATLASS_DEVICE
    TileStdEstSimplified() {};

    // 
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementY> srcMaxTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        ElementY std_scale_factor
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t m_round = layoutDst.shape(0);

        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementY);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = temp_repeat_size;
        uint32_t repeattimes = CeilDiv(m_actual, temp_repeat_size);

        /*
        template <typename T>
        __aicore__ inline void Sub(const LocalTensor<T>& dstLocal, 
            const LocalTensor<T>& src0Local, 
            const LocalTensor<T>& src1Local, 
            const int32_t& calCount)
        */

        // AscendC::Sub(dstTensor,srcMaxTensor,srcMeanTensor,m_actual);
        // AscendC::PipeBarrier<PIPE_V>();

        

        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Muls(const LocalTensor<T>& dstLocal, 
        const LocalTensor<T>& srcLocal, 
        const T& scalarValue, 
        const int32_t& calCount)
        */

        AscendC::SetMaskCount();
        AscendC::SetVectorMask<ElementY, AscendC::MaskMode::COUNTER>(m_actual);
        AscendC::Muls<ElementY,false>(
            dstTensor,
            srcMaxTensor,
            std_scale_factor,
            AscendC::MASK_PLACEHOLDER,
            1,
            AscendC::UnaryRepeatParams{}
        );
        AscendC::SetMaskNorm();
        AscendC::ResetMask();

        AscendC::PipeBarrier<PIPE_V>();
    }
};


template <>
struct TileStdEstSimplified<Arch::AtlasA2,
                Gemm::GemmType<half, layout::VectorLayout>,
                Gemm::GemmType<half, layout::VectorLayout>,
                void>
{
    using ElementX = half;
    using ElementY = half;
    // using ElementY = half;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementX, ElementY>::ElementAccumulator;

    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementX);
    static constexpr uint32_t OUT_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);
    // Mehtods

    CATLASS_DEVICE
    TileStdEstSimplified() {};

    // 
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementY> srcMaxTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        ElementY std_scale_factor
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t m_round = layoutDst.shape(0);

        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementY);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = temp_repeat_size;
        uint32_t repeattimes = CeilDiv(m_actual, temp_repeat_size);

        /*
        template <typename T>
        __aicore__ inline void Sub(const LocalTensor<T>& dstLocal, 
            const LocalTensor<T>& src0Local, 
            const LocalTensor<T>& src1Local, 
            const int32_t& calCount)
        */

        // AscendC::Sub(dstTensor,srcMaxTensor,srcMeanTensor,m_actual);
        // AscendC::PipeBarrier<PIPE_V>();

        

        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Muls(const LocalTensor<T>& dstLocal, 
        const LocalTensor<T>& srcLocal, 
        const T& scalarValue, 
        const int32_t& calCount)
        */

        AscendC::SetMaskCount();
        AscendC::SetVectorMask<ElementY, AscendC::MaskMode::COUNTER>(m_actual);
        AscendC::Muls<ElementY,false>(
            dstTensor,
            srcMaxTensor,
            std_scale_factor,
            AscendC::MASK_PLACEHOLDER,
            1,
            AscendC::UnaryRepeatParams{}
        );
        AscendC::SetMaskNorm();
        AscendC::ResetMask();

        AscendC::PipeBarrier<PIPE_V>();
    }
};

template <>
struct TileStdEstSimplified<Arch::AtlasA2,
                Gemm::GemmType<half, layout::VectorLayout>,
                Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementX = half;
    using ElementY = float;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementX, ElementX>::ElementAccumulator;

    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementX);
    static constexpr uint32_t OUT_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);
    // Mehtods

    CATLASS_DEVICE
    TileStdEstSimplified() {};

    // 
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementX> srcMaxTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        ElementY std_scale_factor
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t m_round = layoutDst.shape(0);

        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementX);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = temp_repeat_size;
        uint32_t repeattimes = CeilDiv(m_actual, temp_repeat_size);

        /*
        template <typename T>
        __aicore__ inline void Sub(const LocalTensor<T>& dstLocal, 
            const LocalTensor<T>& src0Local, 
            const LocalTensor<T>& src1Local, 
            const int32_t& calCount)
        */

        // AscendC::Sub(srcMaxTensor,srcMaxTensor,srcMeanTensor,m_actual);
        // AscendC::PipeBarrier<PIPE_V>();

        AscendC::UnaryRepeatParams castparams;
        castparams.dstBlkStride = 1;
        castparams.srcBlkStride = 1;
        castparams.dstRepStride = 8;
        castparams.srcRepStride = 4;

        /*
        template <typename T1, typename T2>
        __aicore__ inline void Cast(const LocalTensor<T1>& dstLocal, 
            const LocalTensor<T2>& srcLocal, 
            const RoundMode& round_mode, const uint32_t calCount)

        (uint64_t)mask, repeattimes, castparams
        */
        AscendC::Cast<ElementY, ElementX>(
            dstTensor,
            srcMaxTensor,
            AscendC::RoundMode::CAST_NONE, m_actual);
        AscendC::PipeBarrier<PIPE_V>();

        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Muls(const LocalTensor<T>& dstLocal, 
        const LocalTensor<T>& srcLocal, 
        const T& scalarValue, 
        const int32_t& calCount)
        */

        AscendC::SetMaskCount();
        AscendC::SetVectorMask<ElementY, AscendC::MaskMode::COUNTER>(m_actual);
        AscendC::Muls<ElementY,false>(
            dstTensor,
            dstTensor,
            std_scale_factor,
            AscendC::MASK_PLACEHOLDER,
            1,
            AscendC::UnaryRepeatParams{}
        );
        AscendC::SetMaskNorm();
        AscendC::ResetMask();

        AscendC::PipeBarrier<PIPE_V>();
    }
};

/*
将计算std的过程矩阵化，即一次计算整个矩阵中每行的std
*/
template <
    class ElementY
>
struct TileStdEstRobust<Arch::AtlasA2,
                Gemm::GemmType<ElementY, layout::RowMajor>,
                Gemm::GemmType<ElementY, layout::RowMajor>,
                void>
{
    using ElementX = ElementY;
    // using ElementY = half;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementX, ElementY>::ElementAccumulator;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementX);
    static constexpr uint32_t OUT_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);
    // Mehtods

    CATLASS_DEVICE
    TileStdEstRobust() {};

    // 
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementY> srcMeanTensor,
        AscendC::LocalTensor<ElementY> srcMaxTensor,
        AscendC::LocalTensor<ElementY> srcMinTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);

        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementY);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = temp_repeat_size;

        uint64_t sub_mask = elem_repeat_size;
        uint64_t mul_mask = elem_repeat_size;
        uint64_t abs_mask = elem_repeat_size;
        uint64_t sqrt_mask = elem_repeat_size;

        uint32_t repeat_num = n_actual / elem_repeat_size;
        uint32_t remain = n_actual % elem_repeat_size;

        AscendC::BinaryRepeatParams std_params;
        std_params.dstBlkStride = 1;
        std_params.src0BlkStride = 1;
        std_params.src1BlkStride = 1;
        std_params.dstRepStride = RoundUp(n_round, elem_repeat_size) / ELE_NUM_PER_C0;
        std_params.src0RepStride = RoundUp(n_round, elem_repeat_size) / ELE_NUM_PER_C0;
        std_params.src1RepStride = RoundUp(n_round, elem_repeat_size) / ELE_NUM_PER_C0;

        AscendC::UnaryRepeatParams final_params;

        final_params.dstBlkStride = 1;
        final_params.srcBlkStride = 1;
        final_params.dstRepStride = RoundUp(n_round, elem_repeat_size) / ELE_NUM_PER_C0;
        final_params.srcRepStride = RoundUp(n_round, elem_repeat_size) / ELE_NUM_PER_C0;

        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Sub(
            const LocalTensor<T>& dst, 
            const LocalTensor<T>& src0, 
            const LocalTensor<T>& src1, 
            uint64_t mask, 
            const uint8_t repeatTime, 
            const BinaryRepeatParams& repeatParams)
        */

        for(uint32_t i=0; i < repeat_num; i++){
            uint32_t offset = i * elem_repeat_size;
            // AscendC::Add<ElementA, true>
            /*
            AscendC::Add<ElementA, true>(
                srcTensor_m,
                srcTensor_m[offset],
                srcTensor_m,
                add_mask,
                m_actual,
                params);
            */
            AscendC::Sub<ElementY, true>(
                srcMaxTensor[offset], 
                srcMaxTensor[offset], 
                srcMeanTensor[offset], 
                sub_mask,
                m_actual,
                std_params
            );

            // AscendC::Sub(srcMinTensor, srcMeanTensor, srcMinTensor, m_actual);
            AscendC::Sub<ElementY, true>(
                srcMinTensor[offset], 
                srcMeanTensor[offset], 
                srcMinTensor[offset], 
                sub_mask,
                m_actual,
                std_params
            );
        }

        AscendC::PipeBarrier<PIPE_V>();

        if (remain > 0)
        {
            uint32_t offset = repeat_num * elem_repeat_size;
            if (offset + remain > n_actual)
            {
                remain = n_actual - offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;
            AscendC::Sub<ElementY, true>(
                srcMaxTensor[offset], 
                srcMaxTensor[offset], 
                srcMeanTensor[offset], 
                remain_mask,
                m_actual,
                std_params
            );

            // AscendC::Sub(srcMinTensor, srcMeanTensor, srcMinTensor, m_actual);
            AscendC::Sub<ElementY, true>(
                srcMinTensor[offset], 
                srcMeanTensor[offset], 
                srcMinTensor[offset], 
                remain_mask,
                m_actual,
                std_params
            );
        }
         
        AscendC::PipeBarrier<PIPE_V>();
        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Mul(
            const LocalTensor<T>& dst, 
            const LocalTensor<T>& src0, 
            const LocalTensor<T>& src1, 
            uint64_t mask, 
            const uint8_t repeatTime, 
            const BinaryRepeatParams& repeatParams)
        */

        for(uint32_t i=0; i < repeat_num; i++){
            uint32_t offset = i * elem_repeat_size;
            // AscendC::Add<ElementA, true>
            /*
            AscendC::Mul(srcMaxTensor, srcMaxTensor, srcMinTensor, m_actual);
            */
            AscendC::Mul<ElementY, true>(
                srcMaxTensor[offset], 
                srcMaxTensor[offset], 
                srcMinTensor[offset], 
                mul_mask,
                m_actual,
                std_params
            );
        }

        AscendC::PipeBarrier<PIPE_V>();

        if (remain > 0)
        {
            uint32_t offset = repeat_num * elem_repeat_size;
            if (offset + remain > n_actual)
            {
                remain = n_actual - offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;
            AscendC::Mul<ElementY, true>(
                srcMaxTensor[offset], 
                srcMaxTensor[offset], 
                srcMinTensor[offset], 
                remain_mask,
                m_actual,
                std_params
            );
        }

        AscendC::PipeBarrier<PIPE_V>();

        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Abs(
            const LocalTensor<T>& dst, 
            const LocalTensor<T>& src, 
            uint64_t mask, 
            const uint8_t repeatTime, 
            const UnaryRepeatParams& repeatParams)
        */

        for(uint32_t i=0; i < repeat_num; i++){
            uint32_t offset = i * elem_repeat_size;
            // AscendC::Add<ElementA, true>
            /*
            AscendC::Abs(srcMaxTensor, srcMaxTensor, m_actual);
            */
            AscendC::Abs<ElementY, true>(
                srcMaxTensor[offset], 
                srcMaxTensor[offset], 
                abs_mask,
                m_actual,
                final_params
            );
        }

        AscendC::PipeBarrier<PIPE_V>();

        if (remain > 0)
        {
            uint32_t offset = repeat_num * elem_repeat_size;
            if (offset + remain > n_actual)
            {
                remain = n_actual - offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;
            AscendC::Abs<ElementY, true>(
                srcMaxTensor[offset], 
                srcMaxTensor[offset], 
                remain_mask,
                m_actual,
                final_params
            );
        }

        AscendC::PipeBarrier<PIPE_V>();

        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Sqrt(
            const LocalTensor<T>& dst, 
            const LocalTensor<T>& src, 
            uint64_t mask, 
            const uint8_t repeatTime, 
            const UnaryRepeatParams& repeatParams)
        */

        for(uint32_t i=0; i < repeat_num; i++){
            uint32_t offset = i * elem_repeat_size;
            // AscendC::Add<ElementA, true>
            /*
            AscendC::Sqrt(dstTensor, srcMaxTensor, m_actual);
            */
            AscendC::Sqrt<ElementY, true>(
                dstTensor[offset], 
                srcMaxTensor[offset], 
                sqrt_mask,
                m_actual,
                final_params
            );
        }

        AscendC::PipeBarrier<PIPE_V>();

        if (remain > 0)
        {
            uint32_t offset = repeat_num * elem_repeat_size;
            if (offset + remain > n_actual)
            {
                remain = n_actual - offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;
            AscendC::Sqrt<ElementY, true>(
                dstTensor[offset], 
                srcMaxTensor[offset], 
                remain_mask,
                m_actual,
                final_params
            );
        }

        AscendC::PipeBarrier<PIPE_V>();
    }
};

template <>
struct TileStdEstRobust<Arch::AtlasA2,
                Gemm::GemmType<float, layout::RowMajor>,
                Gemm::GemmType<float, layout::RowMajor>,
                void>
{
    using ElementX = float;
    using ElementY = float;
    // using ElementY = half;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementX, ElementY>::ElementAccumulator;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementX);
    static constexpr uint32_t OUT_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);
    // Mehtods

    CATLASS_DEVICE
    TileStdEstRobust() {};

    // 
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementY> srcMeanTensor,
        AscendC::LocalTensor<ElementY> srcMaxTensor,
        AscendC::LocalTensor<ElementY> srcMinTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);

        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementY);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = temp_repeat_size;

        uint64_t sub_mask = elem_repeat_size;
        uint64_t mul_mask = elem_repeat_size;
        uint64_t abs_mask = elem_repeat_size;
        uint64_t sqrt_mask = elem_repeat_size;

        uint32_t repeat_num = n_actual / elem_repeat_size;
        uint32_t remain = n_actual % elem_repeat_size;

        AscendC::BinaryRepeatParams std_params;
        std_params.dstBlkStride = 1;
        std_params.src0BlkStride = 1;
        std_params.src1BlkStride = 1;
        std_params.dstRepStride = RoundUp(n_round, elem_repeat_size) / ELE_NUM_PER_C0;
        std_params.src0RepStride = RoundUp(n_round, elem_repeat_size) / ELE_NUM_PER_C0;
        std_params.src1RepStride = RoundUp(n_round, elem_repeat_size) / ELE_NUM_PER_C0;

        AscendC::UnaryRepeatParams final_params;

        final_params.dstBlkStride = 1;
        final_params.srcBlkStride = 1;
        final_params.dstRepStride = RoundUp(n_round, elem_repeat_size) / ELE_NUM_PER_C0;
        final_params.srcRepStride = RoundUp(n_round, elem_repeat_size) / ELE_NUM_PER_C0;

        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Sub(
            const LocalTensor<T>& dst, 
            const LocalTensor<T>& src0, 
            const LocalTensor<T>& src1, 
            uint64_t mask, 
            const uint8_t repeatTime, 
            const BinaryRepeatParams& repeatParams)
        */

        for(uint32_t i=0; i < repeat_num; i++){
            uint32_t offset = i * elem_repeat_size;
            // AscendC::Add<ElementA, true>
            /*
            AscendC::Add<ElementA, true>(
                srcTensor_m,
                srcTensor_m[offset],
                srcTensor_m,
                add_mask,
                m_actual,
                params);
            */
            AscendC::Sub<ElementY, true>(
                srcMaxTensor[offset], 
                srcMaxTensor[offset], 
                srcMeanTensor[offset], 
                sub_mask,
                m_actual,
                std_params
            );

            // AscendC::Sub(srcMinTensor, srcMeanTensor, srcMinTensor, m_actual);
            AscendC::Sub<ElementY, true>(
                srcMinTensor[offset], 
                srcMeanTensor[offset], 
                srcMinTensor[offset], 
                sub_mask,
                m_actual,
                std_params
            );
        }

        AscendC::PipeBarrier<PIPE_V>();

        if (remain > 0)
        {
            uint32_t offset = repeat_num * elem_repeat_size;
            if (offset + remain > n_actual)
            {
                remain = n_actual - offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;
            AscendC::Sub<ElementY, true>(
                srcMaxTensor[offset], 
                srcMaxTensor[offset], 
                srcMeanTensor[offset], 
                remain_mask,
                m_actual,
                std_params
            );

            // AscendC::Sub(srcMinTensor, srcMeanTensor, srcMinTensor, m_actual);
            AscendC::Sub<ElementY, true>(
                srcMinTensor[offset], 
                srcMeanTensor[offset], 
                srcMinTensor[offset], 
                remain_mask,
                m_actual,
                std_params
            );

            AscendC::PipeBarrier<PIPE_V>();
        }
         
        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Mul(
            const LocalTensor<T>& dst, 
            const LocalTensor<T>& src0, 
            const LocalTensor<T>& src1, 
            uint64_t mask, 
            const uint8_t repeatTime, 
            const BinaryRepeatParams& repeatParams)
        */

        for(uint32_t i=0; i < repeat_num; i++){
            uint32_t offset = i * elem_repeat_size;
            // AscendC::Add<ElementA, true>
            /*
            AscendC::Mul(srcMaxTensor, srcMaxTensor, srcMinTensor, m_actual);
            */
            AscendC::Mul<ElementY, true>(
                srcMaxTensor[offset], 
                srcMaxTensor[offset], 
                srcMinTensor[offset], 
                mul_mask,
                m_actual,
                std_params
            );
        }

        AscendC::PipeBarrier<PIPE_V>();

        if (remain > 0)
        {
            uint32_t offset = repeat_num * elem_repeat_size;
            if (offset + remain > n_actual)
            {
                remain = n_actual - offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;
            AscendC::Mul<ElementY, true>(
                srcMaxTensor[offset], 
                srcMaxTensor[offset], 
                srcMinTensor[offset], 
                remain_mask,
                m_actual,
                std_params
            );

            AscendC::PipeBarrier<PIPE_V>();
        }

        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Abs(
            const LocalTensor<T>& dst, 
            const LocalTensor<T>& src, 
            uint64_t mask, 
            const uint8_t repeatTime, 
            const UnaryRepeatParams& repeatParams)
        */

        for(uint32_t i=0; i < repeat_num; i++){
            uint32_t offset = i * elem_repeat_size;
            // AscendC::Add<ElementA, true>
            /*
            AscendC::Abs(srcMaxTensor, srcMaxTensor, m_actual);
            */
            AscendC::Abs<ElementY, true>(
                srcMaxTensor[offset], 
                srcMaxTensor[offset], 
                abs_mask,
                m_actual,
                final_params
            );
        }

        AscendC::PipeBarrier<PIPE_V>();

        if (remain > 0)
        {
            uint32_t offset = repeat_num * elem_repeat_size;
            if (offset + remain > n_actual)
            {
                remain = n_actual - offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;
            AscendC::Abs<ElementY, true>(
                srcMaxTensor[offset], 
                srcMaxTensor[offset], 
                remain_mask,
                m_actual,
                final_params
            );

            AscendC::PipeBarrier<PIPE_V>();
        }

        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Sqrt(
            const LocalTensor<T>& dst, 
            const LocalTensor<T>& src, 
            uint64_t mask, 
            const uint8_t repeatTime, 
            const UnaryRepeatParams& repeatParams)
        */

        for(uint32_t i=0; i < repeat_num; i++){
            uint32_t offset = i * elem_repeat_size;
            // AscendC::Add<ElementA, true>
            /*
            AscendC::Sqrt(dstTensor, srcMaxTensor, m_actual);
            */
            AscendC::Sqrt<ElementY, true>(
                dstTensor[offset], 
                srcMaxTensor[offset], 
                sqrt_mask,
                m_actual,
                final_params
            );
        }

        AscendC::PipeBarrier<PIPE_V>();

        if (remain > 0)
        {
            uint32_t offset = repeat_num * elem_repeat_size;
            if (offset + remain > n_actual)
            {
                remain = n_actual - offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;
            AscendC::Sqrt<ElementY, true>(
                dstTensor[offset], 
                srcMaxTensor[offset], 
                remain_mask,
                m_actual,
                final_params
            );
        }

        AscendC::PipeBarrier<PIPE_V>();
    }
};


}

#endif