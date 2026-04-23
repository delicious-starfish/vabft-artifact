#ifndef CATLASS_GEMV_TILE_TILE_FAULT_SUM_HPP_SELF
#define CATLASS_GEMV_TILE_TILE_FAULT_SUM_HPP_SELF

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
    Gemv::helper::FT_REDUCE_TYPE REDUCE_TYPE_,  
    class AType,
    class YType,
    class BiasType = void
>
struct TileFaultSum
{
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported TileFaultSum, can not find the specialization.");
};

template <
    /// Tag indicating architecture
    class ArchTag,
    Gemv::helper::FT_REDUCE_TYPE REDUCE_TYPE_,  
    class AType,
    class XType,
    class YType,
    class BiasType = void
>
struct TileFaultSumVmad
{
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported TileFaultSumVmad, can not find the specialization.");
};

template <
    class ElementA,
    class ElementY
>
struct TileFaultSum<Arch::AtlasA2,
                Gemv::helper::FT_REDUCE_TYPE::SUM,
                Gemm::GemmType<ElementA, layout::RowMajor>,
                Gemm::GemmType<ElementY, layout::VectorLayout>,
                void>
{
    // using ElementAccumulator = ElementY;

    // using LayoutDst = layout::RowMajor;
    // using LayoutSrc = layout::RowMajor;
    // static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // // Mehtods

    // CATLASS_DEVICE
    // TileFaultSum() {};

    // CATLASS_DEVICE
    // void operator()(
    //     AscendC::LocalTensor<ElementY> dstTensor,
    //     AscendC::LocalTensor<ElementX> srcTensor_v,
    //     AscendC::LocalTensor<ElementA> srcTensor_m,
    //     AscendC::LocalTensor<ElementAccumulator> temp,
    //     LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    // )
    // {
    //     uint32_t m_actual = layoutSrc.shape(0);
    //     uint32_t n_actual = layoutSrc.shape(1);
    //     uint32_t m_round = layoutDst.shape(0);
    //     uint32_t n_round = layoutDst.shape(1);
    //     uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementAccumulator);
    //     uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
    //     uint32_t mask = temp_repeat_size;
    //     uint32_t repeattimes = CeilDiv(m_actual, temp_repeat_size);
    //     AscendC::Duplicate<ElementAccumulator>(
    //         temp,
    //         (ElementAccumulator)0.0,
    //         temp_repeat_size,
    //         CeilDiv(m_round * temp_repeat_size, temp_repeat_size),
    //         1,
    //         8
    //     );

    //     uint32_t repeat_num = n_actual / temp_repeat_size;
    //     uint32_t remain = n_actual % temp_repeat_size;

    //     AscendC::PipeBarrier<PIPE_V>();
    //     AscendC::BinaryRepeatParams params;
    //     params.dstBlkStride = 1;
    //     params.src0BlkStride = 1;
    //     params.src1BlkStride = 1;
    //     params.dstRepStride = RoundUp(temp_repeat_size, temp_repeat_size) / (BYTE_PER_C0 / sizeof(ElementAccumulator));
    //     params.src0RepStride = RoundUp(n_round, elem_repeat_size) / ELE_NUM_PER_C0;
    //     params.src1RepStride = 0;
    //     AscendC::SetMaskCount();
    //     AscendC::SetVectorMask<ElementAccumulator, AscendC::MaskMode::COUNTER>(m_actual * temp_repeat_size);
    //     for (uint32_t i = 0; i < repeat_num; i++)
    //     {
    //         uint32_t offset = i * temp_repeat_size;
    //         AscendC::MulAddDst<ElementAccumulator, ElementA, false>(
    //             temp,
    //             srcTensor_m[offset],
    //             srcTensor_v[offset],
    //             AscendC::MASK_PLACEHOLDER,
    //             1,
    //             params);

    //         AscendC::PipeBarrier<PIPE_V>();
    //     }
    //     AscendC::SetMaskNorm();
    //     AscendC::ResetMask();

    //     if (remain > 0)
    //     {
    //         uint32_t offset = repeat_num * temp_repeat_size;
    //         if (offset + remain > n_round)
    //         {
    //             remain = n_round - offset;
    //         }
    //         uint64_t remain_mask = remain;
    //         AscendC::MulAddDst<ElementAccumulator, ElementA, true>(
    //             temp,
    //             srcTensor_m[offset],
    //             srcTensor_v[offset],
    //             remain_mask,
    //             m_actual,
    //             params);
    //     }

    //     uint64_t reduce_mask = (repeat_num == 0) ? remain : temp_repeat_size;
    //     AscendC::PipeBarrier<PIPE_V>();
    //     AscendC::WholeReduceSum<ElementAccumulator, true>(
    //         temp,
    //         temp,
    //         reduce_mask,
    //         m_actual,
    //         1,
    //         1,
    //         8);
    //     AscendC::PipeBarrier<PIPE_V>();
    //     AscendC::UnaryRepeatParams castparams;
    //     castparams.dstBlkStride = 1;
    //     castparams.srcBlkStride = 1;
    //     castparams.dstRepStride = 4;
    //     castparams.srcRepStride = 8;
    //     AscendC::Cast<ElementA, ElementAccumulator, true>(
    //         srcTensor_m,
    //         temp,
    //         AscendC::RoundMode::CAST_NONE,
    //         (uint64_t)mask,
    //         repeattimes,
    //         castparams);
    //     AscendC::PipeBarrier<PIPE_V>();

    //     uint64_t add_mask = (m_actual < elem_repeat_size) ? m_actual : elem_repeat_size;
    //     params.dstRepStride = 8;
    //     params.src0RepStride = 8;
    //     params.src1RepStride = 8;
    //     AscendC::Add<ElementA, true>(
    //         dstTensor,
    //         srcTensor_m,
    //         dstTensor,
    //         (uint64_t)add_mask,
    //         CeilDiv(m_round, elem_repeat_size),
    //         params);
    // }
};

template <
    class ElementA,
    class ElementX,
    class ElementY
>
struct TileFaultSumVmad<Arch::AtlasA2,
                Gemv::helper::FT_REDUCE_TYPE::SUM_VMAD,
                Gemm::GemmType<ElementA, layout::RowMajor>,
                Gemm::GemmType<ElementX, layout::VectorLayout>,
                Gemm::GemmType<ElementY, layout::VectorLayout>,
                void>
{
    // using ElementAccumulator = ElementY;

    // using LayoutDst = layout::RowMajor;
    // using LayoutSrc = layout::RowMajor;
    // static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // // Mehtods

    // CATLASS_DEVICE
    // TileFaultSum() {};

    // CATLASS_DEVICE
    // void operator()(
    //     AscendC::LocalTensor<ElementY> dstTensor,
    //     AscendC::LocalTensor<ElementX> srcTensor_v,
    //     AscendC::LocalTensor<ElementA> srcTensor_m,
    //     AscendC::LocalTensor<ElementAccumulator> temp,
    //     LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    // )
    // {
    //     uint32_t m_actual = layoutSrc.shape(0);
    //     uint32_t n_actual = layoutSrc.shape(1);
    //     uint32_t m_round = layoutDst.shape(0);
    //     uint32_t n_round = layoutDst.shape(1);
    //     uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementAccumulator);
    //     uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
    //     uint32_t mask = temp_repeat_size;
    //     uint32_t repeattimes = CeilDiv(m_actual, temp_repeat_size);
    //     AscendC::Duplicate<ElementAccumulator>(
    //         temp,
    //         (ElementAccumulator)0.0,
    //         temp_repeat_size,
    //         CeilDiv(m_round * temp_repeat_size, temp_repeat_size),
    //         1,
    //         8
    //     );

    //     uint32_t repeat_num = n_actual / temp_repeat_size;
    //     uint32_t remain = n_actual % temp_repeat_size;

    //     AscendC::PipeBarrier<PIPE_V>();
    //     AscendC::BinaryRepeatParams params;
    //     params.dstBlkStride = 1;
    //     params.src0BlkStride = 1;
    //     params.src1BlkStride = 1;
    //     params.dstRepStride = RoundUp(temp_repeat_size, temp_repeat_size) / (BYTE_PER_C0 / sizeof(ElementAccumulator));
    //     params.src0RepStride = RoundUp(n_round, elem_repeat_size) / ELE_NUM_PER_C0;
    //     params.src1RepStride = 0;
    //     AscendC::SetMaskCount();
    //     AscendC::SetVectorMask<ElementAccumulator, AscendC::MaskMode::COUNTER>(m_actual * temp_repeat_size);
    //     for (uint32_t i = 0; i < repeat_num; i++)
    //     {
    //         uint32_t offset = i * temp_repeat_size;
    //         AscendC::MulAddDst<ElementAccumulator, ElementA, false>(
    //             temp,
    //             srcTensor_m[offset],
    //             srcTensor_v[offset],
    //             AscendC::MASK_PLACEHOLDER,
    //             1,
    //             params);

    //         AscendC::PipeBarrier<PIPE_V>();
    //     }
    //     AscendC::SetMaskNorm();
    //     AscendC::ResetMask();

    //     if (remain > 0)
    //     {
    //         uint32_t offset = repeat_num * temp_repeat_size;
    //         if (offset + remain > n_round)
    //         {
    //             remain = n_round - offset;
    //         }
    //         uint64_t remain_mask = remain;
    //         AscendC::MulAddDst<ElementAccumulator, ElementA, true>(
    //             temp,
    //             srcTensor_m[offset],
    //             srcTensor_v[offset],
    //             remain_mask,
    //             m_actual,
    //             params);
    //     }

    //     uint64_t reduce_mask = (repeat_num == 0) ? remain : temp_repeat_size;
    //     AscendC::PipeBarrier<PIPE_V>();
    //     AscendC::WholeReduceSum<ElementAccumulator, true>(
    //         temp,
    //         temp,
    //         reduce_mask,
    //         m_actual,
    //         1,
    //         1,
    //         8);
    //     AscendC::PipeBarrier<PIPE_V>();
    //     AscendC::UnaryRepeatParams castparams;
    //     castparams.dstBlkStride = 1;
    //     castparams.srcBlkStride = 1;
    //     castparams.dstRepStride = 4;
    //     castparams.srcRepStride = 8;
    //     AscendC::Cast<ElementA, ElementAccumulator, true>(
    //         srcTensor_m,
    //         temp,
    //         AscendC::RoundMode::CAST_NONE,
    //         (uint64_t)mask,
    //         repeattimes,
    //         castparams);
    //     AscendC::PipeBarrier<PIPE_V>();

    //     uint64_t add_mask = (m_actual < elem_repeat_size) ? m_actual : elem_repeat_size;
    //     params.dstRepStride = 8;
    //     params.src0RepStride = 8;
    //     params.src1RepStride = 8;
    //     AscendC::Add<ElementA, true>(
    //         dstTensor,
    //         srcTensor_m,
    //         dstTensor,
    //         (uint64_t)add_mask,
    //         CeilDiv(m_round, elem_repeat_size),
    //         params);
    // }
};

template <>
struct TileFaultSum<Arch::AtlasA2,
                Gemv::helper::FT_REDUCE_TYPE::SUM,
                Gemm::GemmType<float, layout::RowMajor>,
                Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementA = float;
    using ElementX = float;
    using ElementY = float;
    using ElementAccumulator = ElementY;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // Methods

    CATLASS_DEVICE
    TileFaultSum() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        AscendC::LocalTensor<ElementAccumulator> temp,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = repeat_size;
        uint32_t repeat_num = n_actual / repeat_size;
        uint32_t remain = n_actual % repeat_size;

        // m_actual *
        uint64_t add_mask = repeat_size;

        /*
        控制操作数地址步长的参数。BinaryRepeatParams类型，
        包含操作数相邻迭代间相同datablock的地址步长，
        操作数同一迭代内不同datablock的地址步长等参数。

        相邻迭代间的地址步长参数说明请参考repeatStride；
        同一迭代内datablock的地址步长参数说明请参考dataBlockStride。
        */

        AscendC::BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;
        params.dstRepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        params.src1RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

        

        // AscendC::Duplicate<ElementAccumulator>(
        //     temp,
        //     (ElementAccumulator)0.0,
        //     repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
        //     // 我们的目标是求行和，所以我们的输出空间是行的倍数，且是datablock的倍数
        //     // 因此在这里，我们在赋值0时通过每次内部迭代对应一个输出行的局部和存储与accum空间
        //     CeilDiv(m_actual * repeat_size, repeat_size), // 求行和
        //     1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
        //     8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        // ); // 总的来说，我们开辟
        // AscendC::PipeBarrier<PIPE_V>();
        
        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Add(const LocalTensor<T>& dstLocal, 
        const LocalTensor<T>& src0Local, 
        const LocalTensor<T>& src1Local, 
        uint64_t mask, 
        const uint8_t repeatTimes, 
        const BinaryRepeatParams& repeatParams)
        */

        for (uint32_t i = 1; i < repeat_num; i++) {
            uint32_t offset = i * repeat_size;
            AscendC::Add<ElementA, true>(
                srcTensor_m,
                srcTensor_m[offset],
                srcTensor_m,
                add_mask,
                m_actual,
                params);
            AscendC::PipeBarrier<PIPE_V>();
        }

        if (remain > 0)
        {
            uint32_t offset = repeat_num * repeat_size;
            if (offset + remain > n_actual)
            {
                remain = n_actual - offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;
            AscendC::Add<ElementA, true>(
                srcTensor_m,
                srcTensor_m[offset],
                srcTensor_m,
                remain_mask,
                m_actual,
                params);
        }
        
        uint64_t reduce_mask = (repeat_num == 0) ? remain : repeat_size;
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum<ElementA, true>(
            srcTensor_m,
            srcTensor_m,
            reduce_mask,
            m_actual,
            1,
            1,
            RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0);

        add_mask = (m_actual < repeat_size) ? m_actual : repeat_size;
        params.dstRepStride = 8;
        params.src0RepStride = 8;
        params.src1RepStride = 8;

        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<ElementA, true>(
            dstTensor,
            srcTensor_m,
            dstTensor,
            add_mask,
            CeilDiv(m_round, repeat_size),
            params);
    }
};

template <>
struct TileFaultSum<Arch::AtlasA2,
                Gemv::helper::FT_REDUCE_TYPE::MAX,
                Gemm::GemmType<float, layout::RowMajor>,
                Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementA = float;
    using ElementX = float;
    using ElementY = float;
    using ElementAccumulator = ElementY;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // Methods

    CATLASS_DEVICE
    TileFaultSum() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        AscendC::LocalTensor<ElementAccumulator> temp,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = repeat_size;
        uint32_t repeat_num = n_actual / repeat_size;
        uint32_t remain = n_actual % repeat_size;

        // m_actual *
        uint64_t max_mask = repeat_size;

        /*
        控制操作数地址步长的参数。BinaryRepeatParams类型，
        包含操作数相邻迭代间相同datablock的地址步长，
        操作数同一迭代内不同datablock的地址步长等参数。

        相邻迭代间的地址步长参数说明请参考repeatStride；
        同一迭代内datablock的地址步长参数说明请参考dataBlockStride。
        */

        

        AscendC::BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;
        params.dstRepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        params.src1RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

        // AscendC::Duplicate<ElementAccumulator>(
        //     temp,
        //     (ElementAccumulator)0.0,
        //     repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
        //     // 我们的目标是求行和，所以我们的输出空间是行的倍数，且是datablock的倍数
        //     // 因此在这里，我们在赋值0时通过每次内部迭代对应一个输出行的局部和存储与accum空间
        //     CeilDiv(m_actual * repeat_size, repeat_size), // 求行和
        //     1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
        //     8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        // ); // 总的来说，我们开辟
        // AscendC::PipeBarrier<PIPE_V>();

        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Max(
        const LocalTensor<T>& dstLocal, 
        const LocalTensor<T>& src0Local, 
        const LocalTensor<T>& src1Local, 
        uint64_t mask, const uint8_t repeatTimes, 
        const BinaryRepeatParams& repeatParams)
        */
        
        for (uint32_t i = 1; i < repeat_num; i++) {
            uint32_t offset = i * repeat_size;
            AscendC::Max<ElementA, true>(
                srcTensor_m,
                srcTensor_m[offset],
                srcTensor_m,
                max_mask,
                m_actual,
                params);
            AscendC::PipeBarrier<PIPE_V>();
        }

        if (remain > 0)
        {
            uint32_t offset = repeat_num * repeat_size;
            if (offset + remain > n_actual)
            {
                remain = n_actual - offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;
            AscendC::Max<ElementA, true>(
                srcTensor_m,
                srcTensor_m[offset],
                srcTensor_m,
                remain_mask,
                m_actual,
                params);
        }
        
        uint64_t reduce_mask = (repeat_num == 0) ? remain : repeat_size;
        AscendC::PipeBarrier<PIPE_V>();

        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void WholeReduceMax(const LocalTensor<T>& dstLocal, 
        const LocalTensor<T>& srcLocal, const int32_t mask, 
        const int32_t repeatTimes, const int32_t dstRepStride, 
        const int32_t srcBlkStride, const int32_t srcRepStride, 
        ReduceOrder order = ReduceOrder::ORDER_VALUE_INDEX)
        */

        AscendC::WholeReduceMax<ElementA, true>(
            srcTensor_m,
            srcTensor_m,
            reduce_mask,
            m_actual,
            1,
            1,
            RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE
        );

        max_mask = (m_actual < repeat_size) ? m_actual : repeat_size;
        params.dstRepStride = 8;
        params.src0RepStride = 8;
        params.src1RepStride = 8;

        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Max<ElementA, true>(
            dstTensor,
            srcTensor_m,
            dstTensor,
            max_mask,
            CeilDiv(m_round, repeat_size),
            params);
    }
};

template <>
struct TileFaultSum<Arch::AtlasA2,
                Gemv::helper::FT_REDUCE_TYPE::MAX,
                Gemm::GemmType<half, layout::RowMajor>,
                Gemm::GemmType<half, layout::VectorLayout>,
                void>
{
    using ElementA = half;
    using ElementX = half;
    using ElementY = half;
    using ElementAccumulator = ElementY;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // Methods

    CATLASS_DEVICE
    TileFaultSum() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        AscendC::LocalTensor<ElementAccumulator> temp,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = repeat_size;
        uint32_t repeat_num = n_actual / repeat_size;
        uint32_t remain = n_actual % repeat_size;

        // m_actual *
        uint64_t max_mask = repeat_size;

        /*
        控制操作数地址步长的参数。BinaryRepeatParams类型，
        包含操作数相邻迭代间相同datablock的地址步长，
        操作数同一迭代内不同datablock的地址步长等参数。

        相邻迭代间的地址步长参数说明请参考repeatStride；
        同一迭代内datablock的地址步长参数说明请参考dataBlockStride。
        */

        

        AscendC::BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;
        params.dstRepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        params.src1RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

        // AscendC::Duplicate<ElementAccumulator>(
        //     temp,
        //     (ElementAccumulator)0.0,
        //     repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
        //     // 我们的目标是求行和，所以我们的输出空间是行的倍数，且是datablock的倍数
        //     // 因此在这里，我们在赋值0时通过每次内部迭代对应一个输出行的局部和存储与accum空间
        //     CeilDiv(m_actual * repeat_size, repeat_size), // 求行和
        //     1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
        //     8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        // ); // 总的来说，我们开辟
        // AscendC::PipeBarrier<PIPE_V>();

        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Max(
        const LocalTensor<T>& dstLocal, 
        const LocalTensor<T>& src0Local, 
        const LocalTensor<T>& src1Local, 
        uint64_t mask, const uint8_t repeatTimes, 
        const BinaryRepeatParams& repeatParams)
        */
        
        for (uint32_t i = 1; i < repeat_num; i++) {
            uint32_t offset = i * repeat_size;
            AscendC::Max<ElementA, true>(
                srcTensor_m,
                srcTensor_m[offset],
                srcTensor_m,
                max_mask,
                m_actual,
                params);
            AscendC::PipeBarrier<PIPE_V>();
        }

        if (remain > 0)
        {
            uint32_t offset = repeat_num * repeat_size;
            if (offset + remain > n_actual)
            {
                remain = n_actual - offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;
            AscendC::Max<ElementA, true>(
                srcTensor_m,
                srcTensor_m[offset],
                srcTensor_m,
                remain_mask,
                m_actual,
                params);
        }
        
        uint64_t reduce_mask = (repeat_num == 0) ? remain : repeat_size;
        AscendC::PipeBarrier<PIPE_V>();

        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void WholeReduceMax(const LocalTensor<T>& dstLocal, 
        const LocalTensor<T>& srcLocal, const int32_t mask, 
        const int32_t repeatTimes, const int32_t dstRepStride, 
        const int32_t srcBlkStride, const int32_t srcRepStride, 
        ReduceOrder order = ReduceOrder::ORDER_VALUE_INDEX)
        */

        AscendC::WholeReduceMax<ElementA, true>(
            srcTensor_m,
            srcTensor_m,
            reduce_mask,
            m_actual,
            1,
            1,
            RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE
        );

        max_mask = (m_actual < repeat_size) ? m_actual : repeat_size;
        params.dstRepStride = 8;
        params.src0RepStride = 8;
        params.src1RepStride = 8;

        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Max<ElementA, true>(
            dstTensor,
            srcTensor_m,
            dstTensor,
            max_mask,
            CeilDiv(m_round, repeat_size),
            params);
    }
};

template <>
struct TileFaultSum<Arch::AtlasA2,
                Gemv::helper::FT_REDUCE_TYPE::MAX,
                Gemm::GemmType<half, layout::RowMajor>,
                Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementA = half;
    using ElementX = half;
    using ElementY = float;
    using ElementAccumulator = ElementY;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);
    static constexpr uint32_t DST_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);

    // Methods

    CATLASS_DEVICE
    TileFaultSum() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        AscendC::LocalTensor<ElementA> temp,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t dst_repeat_size = DST_ELE_NUM_PER_C0 * 8;
        uint32_t mask = repeat_size;
        uint32_t repeat_num = n_actual / repeat_size;
        uint32_t remain = n_actual % repeat_size;

        // m_actual *
        uint64_t max_mask = repeat_size;

        /*
        控制操作数地址步长的参数。BinaryRepeatParams类型，
        包含操作数相邻迭代间相同datablock的地址步长，
        操作数同一迭代内不同datablock的地址步长等参数。

        相邻迭代间的地址步长参数说明请参考repeatStride；
        同一迭代内datablock的地址步长参数说明请参考dataBlockStride。
        */

        AscendC::BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;

        // params.dstRepStride = RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0;
        // params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        // params.src1RepStride = RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0;
        
        params.dstRepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        params.src1RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

        // AscendC::Duplicate<ElementAccumulator>(
        //     temp,
        //     (ElementAccumulator)0.0,
        //     repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
        //     // 我们的目标是求行和，所以我们的输出空间是行的倍数，且是datablock的倍数
        //     // 因此在这里，我们在赋值0时通过每次内部迭代对应一个输出行的局部和存储与accum空间
        //     CeilDiv(m_actual * repeat_size, repeat_size), // 求行和
        //     1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
        //     8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        // ); // 总的来说，我们开辟
        // AscendC::PipeBarrier<PIPE_V>();

        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Max(
        const LocalTensor<T>& dstLocal, 
        const LocalTensor<T>& src0Local, 
        const LocalTensor<T>& src1Local, 
        uint64_t mask, const uint8_t repeatTimes, 
        const BinaryRepeatParams& repeatParams)
        */
        
        for (uint32_t i = 1; i < repeat_num; i++) {
            uint32_t offset = i * repeat_size;
            AscendC::Max<ElementA, true>(
                srcTensor_m,
                srcTensor_m[offset],
                srcTensor_m,
                max_mask,
                m_actual,
                params);
            AscendC::PipeBarrier<PIPE_V>();
        }

        if (remain > 0)
        {
            uint32_t offset = repeat_num * repeat_size;
            if (offset + remain > n_actual)
            {
                remain = n_actual - offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;
            AscendC::Max<ElementA, true>(
                srcTensor_m,
                srcTensor_m[offset],
                srcTensor_m,
                remain_mask,
                m_actual,
                params);
        }
        
        uint64_t reduce_mask = (repeat_num == 0) ? remain : repeat_size;
        AscendC::PipeBarrier<PIPE_V>();

        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void WholeReduceMax(const LocalTensor<T>& dstLocal, 
        const LocalTensor<T>& srcLocal, const int32_t mask, 
        const int32_t repeatTimes, const int32_t dstRepStride, 
        const int32_t srcBlkStride, const int32_t srcRepStride, 
        ReduceOrder order = ReduceOrder::ORDER_VALUE_INDEX)
        */

        AscendC::WholeReduceMax<ElementA, true>(
            srcTensor_m,
            srcTensor_m,
            reduce_mask,
            m_actual,
            1,
            1,
            RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE
        );

        AscendC::PipeBarrier<PIPE_V>();

        max_mask = (m_actual < dst_repeat_size) ? m_actual : dst_repeat_size;

        uint32_t dstOffset = m_round;

        AscendC::Cast<ElementY, ElementA>(
            dstTensor[dstOffset],
            srcTensor_m,
            AscendC::RoundMode::CAST_NONE, m_actual);
        AscendC::PipeBarrier<PIPE_V>();

        params.dstRepStride = 8;
        params.src0RepStride = 8;
        params.src1RepStride = 8;

        AscendC::Max<ElementY, true>(
            dstTensor,
            dstTensor[dstOffset],
            dstTensor,
            max_mask,
            CeilDiv(m_round, dst_repeat_size),
            params);
    }
};


// template <>
// struct TileFaultSum<Arch::AtlasA2,
//                 Gemv::helper::FT_REDUCE_TYPE::SUM_MAX_ABE,
//                 Gemm::GemmType<float, layout::RowMajor>,
//                 Gemm::GemmType<float, layout::VectorLayout>,
//                 void>
// {
//     using ElementA = float;
//     using ElementX = float;
//     using ElementY = float;
//     using ElementAccumulator = ElementY;

//     using LayoutDst = layout::RowMajor;
//     using LayoutSrc = layout::RowMajor;

//     static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

//     // Methods

//     CATLASS_DEVICE
//     TileFaultSum() {};

//     CATLASS_DEVICE
//     void operator()(
//         AscendC::LocalTensor<ElementY> dstTensorSum,
//         AscendC::LocalTensor<ElementY> dstTensorMax,
//         AscendC::LocalTensor<ElementA> srcTensor_m,
//         AscendC::LocalTensor<ElementA> sum_workspace,
//         AscendC::LocalTensor<ElementA> max_workspace,
//         LayoutDst const &layoutDst, LayoutSrc const &layoutSrc, uint32_t m_reduce_single
//     )
//     {

//         uint32_t m_actual = layoutSrc.shape(0);
//         uint32_t n_actual = layoutSrc.shape(1);
//         uint32_t m_round = layoutDst.shape(0);
//         uint32_t n_round = layoutDst.shape(1);
        
//         uint32_t m_outer_loop = m_actual / m_reduce_single;
//         uint32_t m_actual_single = m_reduce_single;
//         uint32_t m_actual_remain = m_actual % m_outer_loop;

//         uint32_t m_round_remain = m_round % m_outer_loop;
//         uint32_t m_round_single = m_round / m_outer_loop;

//         uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
//         uint32_t mask = repeat_size;
//         uint32_t repeat_num = n_actual / repeat_size;
//         uint32_t remain = n_actual % repeat_size;

//         // m_actual *
//         uint64_t max_mask = repeat_size;
//         uint64_t add_mask = repeat_size;

//         /*
//         控制操作数地址步长的参数。BinaryRepeatParams类型，
//         包含操作数相邻迭代间相同datablock的地址步长，
//         操作数同一迭代内不同datablock的地址步长等参数。

//         相邻迭代间的地址步长参数说明请参考repeatStride；
//         同一迭代内datablock的地址步长参数说明请参考dataBlockStride。
//         */

//         AscendC::BinaryRepeatParams params;
//         params.dstBlkStride = 1;
//         params.src0BlkStride = 1;
//         params.src1BlkStride = 1;

//         // params.dstRepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

//         params.dstRepStride = RoundUp(repeat_size, repeat_size) / (BYTE_PER_C0 / sizeof(ElementAccumulator));
//         params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
//         params.src1RepStride = RoundUp(repeat_size, repeat_size) / (BYTE_PER_C0 / sizeof(ElementAccumulator));
        
//         // params.src1RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

//         for(uint32_t out_i = 0; out_i < m_outer_loop; out_i++){
//             uint32_t M_actual_offset = out_i * m_actual_single;
//             uint32_t M_round_offset = out_i * m_round_single;

//             uint32_t matrix_offset =  n_round * M_actual_offset;

//             AscendC::Duplicate<ElementAccumulator>(
//                 sum_workspace,
//                 (ElementAccumulator)0.0,
//                 repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
//                 CeilDiv(m_actual_single * repeat_size, repeat_size), // 求行和
//                 1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
//                 8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
//             );
            
//             AscendC::Duplicate<ElementY>(
//                 max_workspace,
//                 (ElementY)0.0,
//                 repeat_size,
//                 CeilDiv(m_actual_single * repeat_size, repeat_size), // 求行最大值
//                 1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
//                 8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
//             );
//             AscendC::PipeBarrier<PIPE_V>();

//             /*
//             template <typename T, bool isSetMask = true>
//             __aicore__ inline void Add(const LocalTensor<T>& dstLocal, 
//                 const LocalTensor<T>& src0Local, const LocalTensor<T>& src1Local, 
//                 uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
//             */
//             for (uint32_t i = 0; i < repeat_num; i++) {
//                 uint32_t offset = matrix_offset + i * repeat_size;
//                 AscendC::Add<ElementAccumulator, true>(
//                     sum_workspace,
//                     srcTensor_m[offset],
//                     sum_workspace,
//                     add_mask,
//                     m_actual_single,
//                     params);
                
//                 AscendC::Max<ElementA, true>(
//                     max_workspace,
//                     srcTensor_m[offset],
//                     max_workspace,
//                     max_mask,
//                     m_actual_single,
//                     params);

//                 AscendC::PipeBarrier<PIPE_V>();
//             }

//             if (remain > 0)
//             {
//                 uint32_t inter_offset = repeat_num * repeat_size;
//                 if (inter_offset + remain > n_actual)
//                 {
//                     remain = n_actual - inter_offset;
//                 }

//                 uint32_t offset = matrix_offset + inter_offset;
//                 // m_actual * 
//                 uint64_t remain_mask = remain;
//                 AscendC::Max<ElementA, true>(
//                     max_workspace,
//                     srcTensor_m[offset],
//                     max_workspace,
//                     remain_mask,
//                     m_actual_single,
//                     params);

//                 AscendC::Add<ElementAccumulator, true>(
//                     sum_workspace,
//                     srcTensor_m[offset],
//                     sum_workspace,
//                     remain_mask,
//                     m_actual_single,
//                     params);
//             }
        
//             uint64_t reduce_mask = (repeat_num == 0) ? remain : repeat_size;
//             AscendC::PipeBarrier<PIPE_V>();

//             AscendC::WholeReduceMax<ElementA, true>(
//                 max_workspace,
//                 max_workspace,
//                 reduce_mask,
//                 m_actual_single,
//                 1,
//                 1,
//                 RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0,
//                 AscendC::ReduceOrder::ORDER_ONLY_VALUE);

//             AscendC::WholeReduceSum<ElementAccumulator, true>(
//                 sum_workspace,
//                 sum_workspace,
//                 reduce_mask,
//                 m_actual_single,
//                 1,
//                 1,
//                 RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0);
            
//             uint64_t add_final_mask = (m_actual_single < repeat_size) ? m_actual_single : repeat_size;
//             uint64_t max_final_mask = (m_actual_single < repeat_size) ? m_actual_single : repeat_size;

//             params.dstRepStride = 8;
//             params.src0RepStride = 8;
//             params.src1RepStride = 8;

//             AscendC::PipeBarrier<PIPE_V>();

//             auto dstMaxTile = dstTensorMax[M_actual_offset];
//             auto dstSumTile = dstTensorSum[M_actual_offset];

//             AscendC::Max<ElementA, true>(
//                 dstMaxTile,
//                 max_workspace,
//                 dstMaxTile,
//                 max_final_mask,
//                 CeilDiv(m_round_single, repeat_size),
//                 params);

//             AscendC::Add<ElementA, true>(
//                 dstSumTile,
//                 sum_workspace,
//                 dstSumTile,
//                 add_final_mask,
//                 CeilDiv(m_round_single, repeat_size),
//                 params);

//             AscendC::PipeBarrier<PIPE_V>();
//         }

//         if(m_actual_remain > 0){

//             params.dstRepStride = RoundUp(repeat_size, repeat_size) / (BYTE_PER_C0 / sizeof(ElementAccumulator));
//             params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
//             params.src1RepStride = RoundUp(repeat_size, repeat_size) / (BYTE_PER_C0 / sizeof(ElementAccumulator));

//             add_mask = repeat_size;
//             max_mask = repeat_size;

//             uint32_t M_actual_offset = m_outer_loop * m_actual_single;
//             uint32_t M_round_offset = m_outer_loop * m_round_single;

//             uint32_t matrix_offset =  n_round * M_actual_offset;

//             AscendC::Duplicate<ElementAccumulator>(
//                 sum_workspace,
//                 (ElementAccumulator)0.0,
//                 repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
//                 CeilDiv(m_actual_remain * repeat_size, repeat_size), // 求行和
//                 1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
//                 8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
//             );
            
//             AscendC::Duplicate<ElementY>(
//                 max_workspace,
//                 (ElementY)0.0,
//                 repeat_size,
//                 CeilDiv(m_actual_remain * repeat_size, repeat_size), // 求行最大值
//                 1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
//                 8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
//             );
//             AscendC::PipeBarrier<PIPE_V>();

//             /*
//             template <typename T, bool isSetMask = true>
//             __aicore__ inline void Add(const LocalTensor<T>& dstLocal, 
//                 const LocalTensor<T>& src0Local, const LocalTensor<T>& src1Local, 
//                 uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
//             */
//             for (uint32_t i = 0; i < repeat_num; i++) {
//                 uint32_t offset = matrix_offset + i * repeat_size;
//                 AscendC::Add<ElementAccumulator, true>(
//                     sum_workspace,
//                     srcTensor_m[offset],
//                     sum_workspace,
//                     add_mask,
//                     m_actual_remain,
//                     params);
                
//                 AscendC::Max<ElementA, true>(
//                     max_workspace,
//                     srcTensor_m[offset],
//                     max_workspace,
//                     max_mask,
//                     m_actual_remain,
//                     params);

//                 AscendC::PipeBarrier<PIPE_V>();
//             }

//             if (remain > 0)
//             {
//                 uint32_t inter_offset = repeat_num * repeat_size;
//                 if (inter_offset + remain > n_actual)
//                 {
//                     remain = n_actual - inter_offset;
//                 }

//                 uint32_t offset = matrix_offset + inter_offset;
//                 // m_actual * 
//                 uint64_t remain_mask = remain;

//                 /*
//                 template <typename T, bool isSetMask = true>
//                 __aicore__ inline void Max(
//                     const LocalTensor<T>& dstLocal, 
//                     const LocalTensor<T>& src0Local, 
//                     const LocalTensor<T>& src1Local, 
//                     uint64_t mask, const uint8_t repeatTimes, 
//                     const BinaryRepeatParams& repeatParams)
//                 */
//                 AscendC::Max<ElementA, true>(
//                     max_workspace,
//                     srcTensor_m[offset],
//                     max_workspace,
//                     remain_mask,
//                     m_actual_remain,
//                     params);

//                 AscendC::Add<ElementAccumulator, true>(
//                     sum_workspace,
//                     srcTensor_m[offset],
//                     sum_workspace,
//                     remain_mask,
//                     m_actual_remain,
//                     params);
//             }
        
//             uint64_t reduce_mask = (repeat_num == 0) ? remain : repeat_size;
//             AscendC::PipeBarrier<PIPE_V>();

//             /*
//             template <typename T, bool isSetMask = true>
//             __aicore__ inline void WholeReduceMax(const LocalTensor<T>& dstLocal, 
//                 const LocalTensor<T>& srcLocal, const int32_t mask, 
//                 const int32_t repeatTimes, const int32_t dstRepStride, 
//                 const int32_t srcBlkStride, const int32_t srcRepStride, 
//                 ReduceOrder order = ReduceOrder::ORDER_VALUE_INDEX)
//             */

//             AscendC::WholeReduceMax<ElementA, true>(
//                 max_workspace,
//                 max_workspace,
//                 reduce_mask,
//                 m_actual_remain,
//                 1,
//                 1,
//                 RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0,
//                 AscendC::ReduceOrder::ORDER_ONLY_VALUE
//             );

//             AscendC::WholeReduceSum<ElementAccumulator, true>(
//                 sum_workspace,
//                 sum_workspace,
//                 reduce_mask,
//                 m_actual_remain,
//                 1,
//                 1,
//                 RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0
//             );
            
//             uint64_t add_final_mask = (m_actual_remain < repeat_size) ? m_actual_remain : repeat_size;
//             uint64_t max_final_mask = (m_actual_remain < repeat_size) ? m_actual_remain : repeat_size;

//             params.dstRepStride = 8;
//             params.src0RepStride = 8;
//             params.src1RepStride = 8;

//             AscendC::PipeBarrier<PIPE_V>();

//             auto dstMaxTile = dstTensorMax[M_actual_offset];
//             auto dstSumTile = dstTensorSum[M_actual_offset];

//             AscendC::Max<ElementA, true>(
//                 dstMaxTile,
//                 max_workspace,
//                 dstMaxTile,
//                 max_final_mask,
//                 CeilDiv(m_round_remain, repeat_size),
//                 params);

//             AscendC::Add<ElementA, true>(
//                 dstSumTile,
//                 sum_workspace,
//                 dstSumTile,
//                 add_final_mask,
//                 CeilDiv(m_round_remain, repeat_size),
//                 params);

//             AscendC::PipeBarrier<PIPE_V>();
//         }

//     }
// };

template <>
struct TileFaultSum<Arch::AtlasA2,
                Gemv::helper::FT_REDUCE_TYPE::SUM_MAX_ABE,
                Gemm::GemmType<float, layout::RowMajor>,
                Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementA = float;
    using ElementX = float;
    using ElementY = float;
    using ElementAccumulator = ElementY;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // Methods

    CATLASS_DEVICE
    TileFaultSum() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensorSum,
        AscendC::LocalTensor<ElementY> dstTensorMax,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        AscendC::LocalTensor<ElementA> tmp_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc, 
        uint32_t m_reduce_single
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);
        
        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = repeat_size;
        uint32_t repeat_num = n_actual / repeat_size;
        uint32_t remain = n_actual % repeat_size;

        // m_actual *
        uint64_t max_mask = repeat_size;
        uint64_t add_mask = repeat_size;

        /*
        控制操作数地址步长的参数。BinaryRepeatParams类型，
        包含操作数相邻迭代间相同datablock的地址步长，
        操作数同一迭代内不同datablock的地址步长等参数。

        相邻迭代间的地址步长参数说明请参考repeatStride；
        同一迭代内datablock的地址步长参数说明请参考dataBlockStride。
        */

        AscendC::BinaryRepeatParams max_params;
        max_params.dstBlkStride = 1;
        max_params.src0BlkStride = 1;
        max_params.src1BlkStride = 1;

        // params.dstRepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

        max_params.dstRepStride = RoundUp(repeat_size, repeat_size) / (BYTE_PER_C0 / sizeof(ElementAccumulator));
        max_params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        max_params.src1RepStride = RoundUp(repeat_size, repeat_size) / (BYTE_PER_C0 / sizeof(ElementAccumulator));
        
        // params.src1RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

        AscendC::Duplicate<ElementY>(
                tmp_workspace,
                (ElementY)0.0,
                repeat_size,
                CeilDiv(m_actual * repeat_size, repeat_size), // 求行最大值
                1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
                8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        );

        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t i = 0; i < repeat_num; i++) {
            uint32_t offset = i * repeat_size;
            AscendC::Max<ElementA, true>(
                tmp_workspace,
                srcTensor_m[offset],
                tmp_workspace,
                max_mask,
                m_actual,
                max_params);

            AscendC::PipeBarrier<PIPE_V>();
        }

        if (remain > 0)
        {
            uint32_t inter_offset = repeat_num * repeat_size;
            if (inter_offset + remain > n_actual)
            {
                remain = n_actual - inter_offset;
            }

            uint32_t offset = inter_offset;
            // m_actual * 
            uint64_t remain_mask = remain;
            AscendC::Max<ElementA, true>(
                tmp_workspace,
                srcTensor_m[offset],
                tmp_workspace,
                remain_mask,
                m_actual,
                max_params);
        }
        
        uint64_t reduce_mask = (repeat_num == 0) ? remain : repeat_size;
        AscendC::PipeBarrier<PIPE_V>();

        max_params.dstRepStride = RoundUp(repeat_size, repeat_size) / (BYTE_PER_C0 / sizeof(ElementAccumulator));
        max_params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        max_params.src1RepStride = RoundUp(repeat_size, repeat_size) / (BYTE_PER_C0 / sizeof(ElementAccumulator));

        AscendC::WholeReduceMax<ElementA, true>(
            tmp_workspace,
            tmp_workspace,
            reduce_mask,
            m_actual,
            1,
            1,
            RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE);

        
        uint64_t max_final_mask = (m_actual < repeat_size) ? m_actual : repeat_size;

        max_params.dstRepStride = 8;
        max_params.src0RepStride = 8;
        max_params.src1RepStride = 8;

        AscendC::PipeBarrier<PIPE_V>();

        
        AscendC::Max<ElementA, true>(
            dstTensorMax,
            tmp_workspace,
            dstTensorMax,
            max_final_mask,
            CeilDiv(m_round, repeat_size),
            max_params);

        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Duplicate<ElementAccumulator>(
                tmp_workspace,
                (ElementAccumulator)0.0,
                repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
                CeilDiv(m_actual * repeat_size, repeat_size), // 求行和
                1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
                8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        );

        AscendC::BinaryRepeatParams sum_params;
        sum_params.dstBlkStride = 1;
        sum_params.src0BlkStride = 1;
        sum_params.src1BlkStride = 1;

        // params.dstRepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

        sum_params.dstRepStride = RoundUp(repeat_size, repeat_size) / (BYTE_PER_C0 / sizeof(ElementAccumulator));
        sum_params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        sum_params.src1RepStride = RoundUp(repeat_size, repeat_size) / (BYTE_PER_C0 / sizeof(ElementAccumulator));

        for (uint32_t i = 0; i < repeat_num; i++) {
            uint32_t offset = i * repeat_size;
            AscendC::Add<ElementAccumulator, true>(
                tmp_workspace,
                srcTensor_m[offset],
                tmp_workspace,
                add_mask,
                m_actual,
                sum_params);
                

            AscendC::PipeBarrier<PIPE_V>();
        }

        if (remain > 0)
        {
            uint32_t inter_offset = repeat_num * repeat_size;
            if (inter_offset + remain > n_actual)
            {
                remain = n_actual - inter_offset;
            }

            uint32_t offset = inter_offset;
            // m_actual * 
            uint64_t remain_mask = remain;

            AscendC::Add<ElementAccumulator, true>(
                tmp_workspace,
                srcTensor_m[offset],
                tmp_workspace,
                remain_mask,
                m_actual,
                sum_params);
        }
        
        reduce_mask = (repeat_num == 0) ? remain : repeat_size;
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::WholeReduceSum<ElementAccumulator, true>(
            tmp_workspace,
            tmp_workspace,
            reduce_mask,
            m_actual,
            1,
            1,
            RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0);
        
        uint64_t add_final_mask = (m_actual < repeat_size) ? m_actual : repeat_size;

        sum_params.dstRepStride = 8;
        sum_params.src0RepStride = 8;
        sum_params.src1RepStride = 8;

        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Add<ElementA, true>(
            dstTensorSum,
            tmp_workspace,
            dstTensorSum,
            add_final_mask,
            CeilDiv(m_round, repeat_size),
            sum_params);

        AscendC::PipeBarrier<PIPE_V>();
    }
};

template <>
struct TileFaultSum<Arch::AtlasA2,
                Gemv::helper::FT_REDUCE_TYPE::SUM_MAX_ABE,
                Gemm::GemmType<half, layout::RowMajor>,
                Gemm::GemmType<half, layout::VectorLayout>,
                void>
{
    using ElementA = half;
    using ElementX = half;
    using ElementY = half;
    using ElementAccumulator = ElementY;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // Methods

    CATLASS_DEVICE
    TileFaultSum() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensorSum,
        AscendC::LocalTensor<ElementY> dstTensorMax,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        AscendC::LocalTensor<ElementA> tmp_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc, 
        uint32_t m_reduce_single
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);
        
        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = repeat_size;
        uint32_t repeat_num = n_actual / repeat_size;
        uint32_t remain = n_actual % repeat_size;

        // m_actual *
        uint64_t max_mask = repeat_size;
        uint64_t add_mask = repeat_size;

        /*
        控制操作数地址步长的参数。BinaryRepeatParams类型，
        包含操作数相邻迭代间相同datablock的地址步长，
        操作数同一迭代内不同datablock的地址步长等参数。

        相邻迭代间的地址步长参数说明请参考repeatStride；
        同一迭代内datablock的地址步长参数说明请参考dataBlockStride。
        */

        AscendC::BinaryRepeatParams max_params;
        max_params.dstBlkStride = 1;
        max_params.src0BlkStride = 1;
        max_params.src1BlkStride = 1;

        // params.dstRepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

        max_params.dstRepStride = RoundUp(repeat_size, repeat_size) / (BYTE_PER_C0 / sizeof(ElementAccumulator));
        max_params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        max_params.src1RepStride = RoundUp(repeat_size, repeat_size) / (BYTE_PER_C0 / sizeof(ElementAccumulator));
        
        // params.src1RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

        AscendC::Duplicate<ElementY>(
                tmp_workspace,
                (ElementY)0.0,
                repeat_size,
                CeilDiv(m_actual * repeat_size, repeat_size), // 求行最大值
                1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
                8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        );

        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t i = 0; i < repeat_num; i++) {
            uint32_t offset = i * repeat_size;
            AscendC::Max<ElementA, true>(
                tmp_workspace,
                srcTensor_m[offset],
                tmp_workspace,
                max_mask,
                m_actual,
                max_params);

            AscendC::PipeBarrier<PIPE_V>();
        }

        if (remain > 0)
        {
            uint32_t inter_offset = repeat_num * repeat_size;
            if (inter_offset + remain > n_actual)
            {
                remain = n_actual - inter_offset;
            }

            uint32_t offset = inter_offset;
            // m_actual * 
            uint64_t remain_mask = remain;
            AscendC::Max<ElementA, true>(
                tmp_workspace,
                srcTensor_m[offset],
                tmp_workspace,
                remain_mask,
                m_actual,
                max_params);
        }
        
        uint64_t reduce_mask = (repeat_num == 0) ? remain : repeat_size;
        AscendC::PipeBarrier<PIPE_V>();

        max_params.dstRepStride = RoundUp(repeat_size, repeat_size) / (BYTE_PER_C0 / sizeof(ElementAccumulator));
        max_params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        max_params.src1RepStride = RoundUp(repeat_size, repeat_size) / (BYTE_PER_C0 / sizeof(ElementAccumulator));

        AscendC::WholeReduceMax<ElementA, true>(
            tmp_workspace,
            tmp_workspace,
            reduce_mask,
            m_actual,
            1,
            1,
            RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE);

        
        uint64_t max_final_mask = (m_actual < repeat_size) ? m_actual : repeat_size;

        max_params.dstRepStride = 8;
        max_params.src0RepStride = 8;
        max_params.src1RepStride = 8;

        AscendC::PipeBarrier<PIPE_V>();

        
        AscendC::Max<ElementA, true>(
            dstTensorMax,
            tmp_workspace,
            dstTensorMax,
            max_final_mask,
            CeilDiv(m_round, repeat_size),
            max_params);

        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Duplicate<ElementAccumulator>(
                tmp_workspace,
                (ElementAccumulator)0.0,
                repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
                CeilDiv(m_actual * repeat_size, repeat_size), // 求行和
                1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
                8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        );

        AscendC::BinaryRepeatParams sum_params;
        sum_params.dstBlkStride = 1;
        sum_params.src0BlkStride = 1;
        sum_params.src1BlkStride = 1;

        // params.dstRepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

        sum_params.dstRepStride = RoundUp(repeat_size, repeat_size) / (BYTE_PER_C0 / sizeof(ElementAccumulator));
        sum_params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        sum_params.src1RepStride = RoundUp(repeat_size, repeat_size) / (BYTE_PER_C0 / sizeof(ElementAccumulator));

        for (uint32_t i = 0; i < repeat_num; i++) {
            uint32_t offset = i * repeat_size;
            AscendC::Add<ElementAccumulator, true>(
                tmp_workspace,
                srcTensor_m[offset],
                tmp_workspace,
                add_mask,
                m_actual,
                sum_params);
                

            AscendC::PipeBarrier<PIPE_V>();
        }

        if (remain > 0)
        {
            uint32_t inter_offset = repeat_num * repeat_size;
            if (inter_offset + remain > n_actual)
            {
                remain = n_actual - inter_offset;
            }

            uint32_t offset = inter_offset;
            // m_actual * 
            uint64_t remain_mask = remain;

            AscendC::Add<ElementAccumulator, true>(
                tmp_workspace,
                srcTensor_m[offset],
                tmp_workspace,
                remain_mask,
                m_actual,
                sum_params);
        }
        
        reduce_mask = (repeat_num == 0) ? remain : repeat_size;
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::WholeReduceSum<ElementAccumulator, true>(
            tmp_workspace,
            tmp_workspace,
            reduce_mask,
            m_actual,
            1,
            1,
            RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0);
        
        uint64_t add_final_mask = (m_actual < repeat_size) ? m_actual : repeat_size;

        sum_params.dstRepStride = 8;
        sum_params.src0RepStride = 8;
        sum_params.src1RepStride = 8;

        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Add<ElementA, true>(
            dstTensorSum,
            tmp_workspace,
            dstTensorSum,
            add_final_mask,
            CeilDiv(m_round, repeat_size),
            sum_params);

        AscendC::PipeBarrier<PIPE_V>();
    }
};

template <>
struct TileFaultSum<Arch::AtlasA2,
                Gemv::helper::FT_REDUCE_TYPE::SUM_MAX_ABE,
                Gemm::GemmType<half, layout::RowMajor>,
                Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementA = half;
    using ElementX = half;
    using ElementY = float;
    using ElementAccumulator = ElementY;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);
    static constexpr uint32_t DST_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);

    // Methods

    CATLASS_DEVICE
    TileFaultSum() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensorSum,
        AscendC::LocalTensor<ElementY> dstTensorMax,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        AscendC::LocalTensor<ElementA> tmp_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc, 
        uint32_t m_reduce_single
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);
        
        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t dst_repeat_size = DST_ELE_NUM_PER_C0 * 8;
        uint32_t mask = repeat_size;
        uint32_t repeat_num = n_actual / repeat_size;
        uint32_t remain = n_actual % repeat_size;

        // m_actual *
        uint64_t max_mask = repeat_size;
        uint64_t add_mask = repeat_size;

        uint32_t dstOffset = m_round;

        /*
        控制操作数地址步长的参数。BinaryRepeatParams类型，
        包含操作数相邻迭代间相同datablock的地址步长，
        操作数同一迭代内不同datablock的地址步长等参数。

        相邻迭代间的地址步长参数说明请参考repeatStride；
        同一迭代内datablock的地址步长参数说明请参考dataBlockStride。
        */

        AscendC::BinaryRepeatParams max_params;
        max_params.dstBlkStride = 1;
        max_params.src0BlkStride = 1;
        max_params.src1BlkStride = 1;

        // params.dstRepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

        max_params.dstRepStride = RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0;
        max_params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        max_params.src1RepStride = RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0;
        
        // params.src1RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

        AscendC::Duplicate<ElementA>(
                tmp_workspace,
                (ElementY)0.0,
                repeat_size,
                CeilDiv(m_actual * repeat_size, repeat_size), // 求行最大值
                1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
                8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        );

        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t i = 0; i < repeat_num; i++) {
            uint32_t offset = i * repeat_size;
            AscendC::Max<ElementA, true>(
                tmp_workspace,
                srcTensor_m[offset],
                tmp_workspace,
                max_mask,
                m_actual,
                max_params);

            AscendC::PipeBarrier<PIPE_V>();
        }

        if (remain > 0)
        {
            uint32_t inter_offset = repeat_num * repeat_size;
            if (inter_offset + remain > n_actual)
            {
                remain = n_actual - inter_offset;
            }

            uint32_t offset = inter_offset;
            // m_actual * 
            uint64_t remain_mask = remain;
            AscendC::Max<ElementA, true>(
                tmp_workspace,
                srcTensor_m[offset],
                tmp_workspace,
                remain_mask,
                m_actual,
                max_params);
        }
        
        uint64_t reduce_mask = (repeat_num == 0) ? remain : repeat_size;
        AscendC::PipeBarrier<PIPE_V>();

        max_params.dstRepStride = RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0;
        max_params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        max_params.src1RepStride = RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0;

        AscendC::WholeReduceMax<ElementA, true>(
            tmp_workspace,
            tmp_workspace,
            reduce_mask,
            m_actual,
            1,
            1,
            RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE);

        AscendC::PipeBarrier<PIPE_V>();

        
        AscendC::Cast<ElementY, ElementA>(
            dstTensorMax[dstOffset],
            tmp_workspace,
            AscendC::RoundMode::CAST_NONE, m_actual);
        AscendC::PipeBarrier<PIPE_V>();

        
        uint64_t max_final_mask = (m_actual < dst_repeat_size) ? m_actual : dst_repeat_size;

        max_params.dstRepStride = 8;
        max_params.src0RepStride = 8;
        max_params.src1RepStride = 8;
        
        AscendC::Max<ElementY, true>(
            dstTensorMax,
            dstTensorMax[dstOffset],
            dstTensorMax,
            max_final_mask,
            CeilDiv(m_round, dst_repeat_size),
            max_params);

        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Duplicate<ElementA>(
                tmp_workspace,
                (ElementA)0.0,
                repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
                CeilDiv(m_actual * repeat_size, repeat_size), // 求行和
                1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
                8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        );

        AscendC::BinaryRepeatParams sum_params;
        sum_params.dstBlkStride = 1;
        sum_params.src0BlkStride = 1;
        sum_params.src1BlkStride = 1;

        // params.dstRepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

        sum_params.dstRepStride = RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0;
        sum_params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        sum_params.src1RepStride = RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0;

        for (uint32_t i = 0; i < repeat_num; i++) {
            uint32_t offset = i * repeat_size;
            AscendC::Add<ElementA, true>(
                tmp_workspace,
                srcTensor_m[offset],
                tmp_workspace,
                add_mask,
                m_actual,
                sum_params);
                

            AscendC::PipeBarrier<PIPE_V>();
        }

        if (remain > 0)
        {
            uint32_t inter_offset = repeat_num * repeat_size;
            if (inter_offset + remain > n_actual)
            {
                remain = n_actual - inter_offset;
            }

            uint32_t offset = inter_offset;
            // m_actual * 
            uint64_t remain_mask = remain;

            AscendC::Add<ElementA, true>(
                tmp_workspace,
                srcTensor_m[offset],
                tmp_workspace,
                remain_mask,
                m_actual,
                sum_params);
        }
        
        reduce_mask = (repeat_num == 0) ? remain : repeat_size;
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::WholeReduceSum<ElementA, true>(
            tmp_workspace,
            tmp_workspace,
            reduce_mask,
            m_actual,
            1,
            1,
            RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0);

        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Cast<ElementY, ElementA>(
            dstTensorSum[dstOffset],
            tmp_workspace,
            AscendC::RoundMode::CAST_NONE, m_actual);
        AscendC::PipeBarrier<PIPE_V>();
        
        uint64_t add_final_mask = (m_actual < dst_repeat_size) ? m_actual : dst_repeat_size;

        sum_params.dstRepStride = 8;
        sum_params.src0RepStride = 8;
        sum_params.src1RepStride = 8;

        AscendC::Add<ElementY, true>(
            dstTensorSum,
            dstTensorSum[dstOffset],
            dstTensorSum,
            add_final_mask,
            CeilDiv(m_round, dst_repeat_size),
            sum_params);

        AscendC::PipeBarrier<PIPE_V>();
    }
};



template <>
struct TileFaultSum<Arch::AtlasA2,
                Gemv::helper::FT_REDUCE_TYPE::SUM_MAX,
                Gemm::GemmType<float, layout::RowMajor>,
                Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementA = float;
    using ElementX = float;
    using ElementY = float;
    using ElementAccumulator = ElementY;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // Methods

    CATLASS_DEVICE
    TileFaultSum() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensorSum,
        AscendC::LocalTensor<ElementY> dstTensorMax,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        AscendC::LocalTensor<ElementAccumulator> sum_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = repeat_size;
        uint32_t repeat_num = n_actual / repeat_size;
        uint32_t remain = n_actual % repeat_size;

        // m_actual *
        uint64_t add_mask = repeat_size;
        uint64_t max_mask = repeat_size;

        /*
        控制操作数地址步长的参数。BinaryRepeatParams类型，
        包含操作数相邻迭代间相同datablock的地址步长，
        操作数同一迭代内不同datablock的地址步长等参数。

        相邻迭代间的地址步长参数说明请参考repeatStride；
        同一迭代内datablock的地址步长参数说明请参考dataBlockStride。
        */

        AscendC::BinaryRepeatParams max_params;
        max_params.dstBlkStride = 1;
        max_params.src0BlkStride = 1;
        max_params.src1BlkStride = 1;
        max_params.dstRepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        max_params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        max_params.src1RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

        AscendC::BinaryRepeatParams sum_params;
        sum_params.dstBlkStride = 1;
        sum_params.src0BlkStride = 1;
        sum_params.src1BlkStride = 1;

        // params.dstRepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

        sum_params.dstRepStride = RoundUp(repeat_size, repeat_size) / (BYTE_PER_C0 / sizeof(ElementAccumulator));
        sum_params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        sum_params.src1RepStride = RoundUp(repeat_size, repeat_size) / (BYTE_PER_C0 / sizeof(ElementAccumulator));

        

        // AscendC::Duplicate<ElementAccumulator>(
        //     temp,
        //     (ElementAccumulator)0.0,
        //     repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
        //     // 我们的目标是求行和，所以我们的输出空间是行的倍数，且是datablock的倍数
        //     // 因此在这里，我们在赋值0时通过每次内部迭代对应一个输出行的局部和存储与accum空间
        //     CeilDiv(m_actual * repeat_size, repeat_size), // 求行和
        //     1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
        //     8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        // ); // 总的来说，我们开辟
        // AscendC::PipeBarrier<PIPE_V>();
        
        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Add(const LocalTensor<T>& dstLocal, 
        const LocalTensor<T>& src0Local, 
        const LocalTensor<T>& src1Local, 
        uint64_t mask, 
        const uint8_t repeatTimes, 
        const BinaryRepeatParams& repeatParams)
        */


        AscendC::Duplicate<ElementAccumulator>(
                sum_workspace,
                (ElementAccumulator)0.0,
                repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
                CeilDiv(m_actual * repeat_size, repeat_size), // 求行和
                1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
                8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        );

        AscendC::PipeBarrier<PIPE_V>();
        
        AscendC::Add<ElementAccumulator, true>(
            sum_workspace,
            srcTensor_m,
            sum_workspace,
            add_mask,
            m_actual,
            sum_params);
        
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t i = 1; i < repeat_num; i++) {
            uint32_t offset = i * repeat_size;

            AscendC::Add<ElementAccumulator, true>(
                sum_workspace,
                srcTensor_m[offset],
                sum_workspace,
                add_mask,
                m_actual,
                sum_params);
            
            AscendC::Max<ElementA, true>(
                srcTensor_m,
                srcTensor_m[offset],
                srcTensor_m,
                max_mask,
                m_actual,
                max_params);
                
            AscendC::PipeBarrier<PIPE_V>();
        }

        if (remain > 0)
        {
            uint32_t offset = repeat_num * repeat_size;
            if (offset + remain > n_actual)
            {
                remain = n_actual - offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;
            AscendC::Add<ElementAccumulator, true>(
                sum_workspace,
                srcTensor_m[offset],
                sum_workspace,
                remain_mask,
                m_actual,
                sum_params);
            
            AscendC::Max<ElementA, true>(
                srcTensor_m,
                srcTensor_m[offset],
                srcTensor_m,
                remain_mask,
                m_actual,
                max_params);
        }
        
        uint64_t reduce_mask = (repeat_num == 0) ? remain : repeat_size;
        AscendC::PipeBarrier<PIPE_V>();
        
        AscendC::WholeReduceMax<ElementA, true>(
            srcTensor_m,
            srcTensor_m,
            reduce_mask,
            m_actual,
            1,
            1,
            RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE);

        AscendC::WholeReduceSum<ElementAccumulator, true>(
            sum_workspace,
            sum_workspace,
            reduce_mask,
            m_actual,
            1,
            1,
            RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0);


        uint64_t add_final_mask = (m_actual < repeat_size) ? m_actual : repeat_size;
        uint64_t max_final_mask = (m_actual < repeat_size) ? m_actual : repeat_size;

        sum_params.dstRepStride = 8;
        sum_params.src0RepStride = 8;
        sum_params.src1RepStride = 8;

        max_params.dstRepStride = 8;
        max_params.src0RepStride = 8;
        max_params.src1RepStride = 8;

        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Add<ElementA, true>(
            dstTensorSum,
            sum_workspace,
            dstTensorSum,
            add_final_mask,
            CeilDiv(m_round, repeat_size),
            sum_params);

        AscendC::Max<ElementA, true>(
            dstTensorMax,
            srcTensor_m,
            dstTensorMax,
            max_final_mask,
            CeilDiv(m_round, repeat_size),
            max_params);
        
    }
};

template <>
struct TileFaultSum<Arch::AtlasA2,
                Gemv::helper::FT_REDUCE_TYPE::SUM_MAX_MIXED,
                Gemm::GemmType<float, layout::RowMajor>,
                Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementA = float;
    using ElementX = float;
    using ElementY = float;
    using ElementAccumulator = ElementY;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // Methods

    CATLASS_DEVICE
    TileFaultSum() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensorSum,
        AscendC::LocalTensor<ElementY> dstTensorMax,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        AscendC::LocalTensor<ElementAccumulator> sum_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = repeat_size;
        uint32_t repeat_num = n_actual / repeat_size;
        uint32_t remain = n_actual % repeat_size;

        // m_actual *
        uint64_t add_mask = repeat_size;
        uint64_t max_mask = repeat_size;

        /*
        控制操作数地址步长的参数。BinaryRepeatParams类型，
        包含操作数相邻迭代间相同datablock的地址步长，
        操作数同一迭代内不同datablock的地址步长等参数。

        相邻迭代间的地址步长参数说明请参考repeatStride；
        同一迭代内datablock的地址步长参数说明请参考dataBlockStride。
        */

        AscendC::BinaryRepeatParams max_params;
        max_params.dstBlkStride = 1;
        max_params.src0BlkStride = 1;
        max_params.src1BlkStride = 1;
        max_params.dstRepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        max_params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        max_params.src1RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

        AscendC::BinaryRepeatParams sum_params;
        sum_params.dstBlkStride = 1;
        sum_params.src0BlkStride = 1;
        sum_params.src1BlkStride = 1;

        // params.dstRepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

        sum_params.dstRepStride = RoundUp(repeat_size, repeat_size) / (BYTE_PER_C0 / sizeof(ElementAccumulator));
        sum_params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        sum_params.src1RepStride = RoundUp(repeat_size, repeat_size) / (BYTE_PER_C0 / sizeof(ElementAccumulator));

        

        // AscendC::Duplicate<ElementAccumulator>(
        //     temp,
        //     (ElementAccumulator)0.0,
        //     repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
        //     // 我们的目标是求行和，所以我们的输出空间是行的倍数，且是datablock的倍数
        //     // 因此在这里，我们在赋值0时通过每次内部迭代对应一个输出行的局部和存储与accum空间
        //     CeilDiv(m_actual * repeat_size, repeat_size), // 求行和
        //     1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
        //     8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        // ); // 总的来说，我们开辟
        // AscendC::PipeBarrier<PIPE_V>();
        
        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Add(const LocalTensor<T>& dstLocal, 
        const LocalTensor<T>& src0Local, 
        const LocalTensor<T>& src1Local, 
        uint64_t mask, 
        const uint8_t repeatTimes, 
        const BinaryRepeatParams& repeatParams)
        */


        AscendC::Duplicate<ElementAccumulator>(
                sum_workspace,
                (ElementAccumulator)0.0,
                repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
                CeilDiv(m_actual * repeat_size, repeat_size), // 求行和
                1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
                8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        );

        AscendC::PipeBarrier<PIPE_V>();
        
        AscendC::Add<ElementAccumulator, true>(
            sum_workspace,
            srcTensor_m,
            sum_workspace,
            add_mask,
            m_actual,
            sum_params);
        
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t i = 1; i < repeat_num; i++) {
            uint32_t offset = i * repeat_size;

            AscendC::Add<ElementAccumulator, true>(
                sum_workspace,
                srcTensor_m[offset],
                sum_workspace,
                add_mask,
                m_actual,
                sum_params);
            
            AscendC::Max<ElementA, true>(
                srcTensor_m,
                srcTensor_m[offset],
                srcTensor_m,
                max_mask,
                m_actual,
                max_params);
                
            AscendC::PipeBarrier<PIPE_V>();
        }

        if (remain > 0)
        {
            uint32_t offset = repeat_num * repeat_size;
            if (offset + remain > n_actual)
            {
                remain = n_actual - offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;
            AscendC::Add<ElementAccumulator, true>(
                sum_workspace,
                srcTensor_m[offset],
                sum_workspace,
                remain_mask,
                m_actual,
                sum_params);
            
            AscendC::Max<ElementA, true>(
                srcTensor_m,
                srcTensor_m[offset],
                srcTensor_m,
                remain_mask,
                m_actual,
                max_params);
        }
        
        uint64_t reduce_mask = (repeat_num == 0) ? remain : repeat_size;
        AscendC::PipeBarrier<PIPE_V>();
        
        AscendC::WholeReduceMax<ElementA, true>(
            srcTensor_m,
            srcTensor_m,
            reduce_mask,
            m_actual,
            1,
            1,
            RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE);

        AscendC::WholeReduceSum<ElementAccumulator, true>(
            sum_workspace,
            sum_workspace,
            reduce_mask,
            m_actual,
            1,
            1,
            RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0);


        uint64_t add_final_mask = (m_actual < repeat_size) ? m_actual : repeat_size;
        uint64_t max_final_mask = (m_actual < repeat_size) ? m_actual : repeat_size;

        sum_params.dstRepStride = 8;
        sum_params.src0RepStride = 8;
        sum_params.src1RepStride = 8;

        max_params.dstRepStride = 8;
        max_params.src0RepStride = 8;
        max_params.src1RepStride = 8;

        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Add<ElementA, true>(
            dstTensorSum,
            sum_workspace,
            dstTensorSum,
            add_final_mask,
            CeilDiv(m_round, repeat_size),
            sum_params);

        AscendC::Max<ElementA, true>(
            dstTensorMax,
            srcTensor_m,
            dstTensorMax,
            max_final_mask,
            CeilDiv(m_round, repeat_size),
            max_params);
        
    }
};


template <>
struct TileFaultSum<Arch::AtlasA2,
                Gemv::helper::FT_REDUCE_TYPE::SUM_MAX,
                Gemm::GemmType<half, layout::RowMajor>,
                Gemm::GemmType<half, layout::VectorLayout>,
                void>
{
    using ElementA = half;
    using ElementX = half;
    using ElementY = half;
    using ElementAccumulator = ElementY;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // Methods

    CATLASS_DEVICE
    TileFaultSum() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensorSum,
        AscendC::LocalTensor<ElementY> dstTensorMax,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        AscendC::LocalTensor<ElementA> sum_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = repeat_size;
        uint32_t repeat_num = n_actual / repeat_size;
        uint32_t remain = n_actual % repeat_size;

        // m_actual *
        uint64_t add_mask = repeat_size;
        uint64_t max_mask = repeat_size;

        /*
        控制操作数地址步长的参数。BinaryRepeatParams类型，
        包含操作数相邻迭代间相同datablock的地址步长，
        操作数同一迭代内不同datablock的地址步长等参数。

        相邻迭代间的地址步长参数说明请参考repeatStride；
        同一迭代内datablock的地址步长参数说明请参考dataBlockStride。
        */

        AscendC::BinaryRepeatParams max_params;
        max_params.dstBlkStride = 1;
        max_params.src0BlkStride = 1;
        max_params.src1BlkStride = 1;
        max_params.dstRepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        max_params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        max_params.src1RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

        AscendC::BinaryRepeatParams sum_params;
        sum_params.dstBlkStride = 1;
        sum_params.src0BlkStride = 1;
        sum_params.src1BlkStride = 1;

        // params.dstRepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

        sum_params.dstRepStride = RoundUp(repeat_size, repeat_size) / (BYTE_PER_C0 / sizeof(ElementAccumulator));
        sum_params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        sum_params.src1RepStride = RoundUp(repeat_size, repeat_size) / (BYTE_PER_C0 / sizeof(ElementAccumulator));

        

        // AscendC::Duplicate<ElementAccumulator>(
        //     temp,
        //     (ElementAccumulator)0.0,
        //     repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
        //     // 我们的目标是求行和，所以我们的输出空间是行的倍数，且是datablock的倍数
        //     // 因此在这里，我们在赋值0时通过每次内部迭代对应一个输出行的局部和存储与accum空间
        //     CeilDiv(m_actual * repeat_size, repeat_size), // 求行和
        //     1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
        //     8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        // ); // 总的来说，我们开辟
        // AscendC::PipeBarrier<PIPE_V>();
        
        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Add(const LocalTensor<T>& dstLocal, 
        const LocalTensor<T>& src0Local, 
        const LocalTensor<T>& src1Local, 
        uint64_t mask, 
        const uint8_t repeatTimes, 
        const BinaryRepeatParams& repeatParams)
        */


        AscendC::Duplicate<ElementAccumulator>(
                sum_workspace,
                (ElementAccumulator)0.0,
                repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
                CeilDiv(m_actual * repeat_size, repeat_size), // 求行和
                1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
                8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        );

        AscendC::PipeBarrier<PIPE_V>();
        
        AscendC::Add<ElementAccumulator, true>(
            sum_workspace,
            srcTensor_m,
            sum_workspace,
            add_mask,
            m_actual,
            sum_params);
        
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t i = 1; i < repeat_num; i++) {
            uint32_t offset = i * repeat_size;

            AscendC::Add<ElementAccumulator, true>(
                sum_workspace,
                srcTensor_m[offset],
                sum_workspace,
                add_mask,
                m_actual,
                sum_params);
            
            AscendC::Max<ElementA, true>(
                srcTensor_m,
                srcTensor_m[offset],
                srcTensor_m,
                max_mask,
                m_actual,
                max_params);
                
            AscendC::PipeBarrier<PIPE_V>();
        }

        if (remain > 0)
        {
            uint32_t offset = repeat_num * repeat_size;
            if (offset + remain > n_actual)
            {
                remain = n_actual - offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;
            AscendC::Add<ElementAccumulator, true>(
                sum_workspace,
                srcTensor_m[offset],
                sum_workspace,
                remain_mask,
                m_actual,
                sum_params);
            
            AscendC::Max<ElementA, true>(
                srcTensor_m,
                srcTensor_m[offset],
                srcTensor_m,
                remain_mask,
                m_actual,
                max_params);
        }
        
        uint64_t reduce_mask = (repeat_num == 0) ? remain : repeat_size;
        AscendC::PipeBarrier<PIPE_V>();
        
        AscendC::WholeReduceMax<ElementA, true>(
            srcTensor_m,
            srcTensor_m,
            reduce_mask,
            m_actual,
            1,
            1,
            RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE);

        AscendC::WholeReduceSum<ElementAccumulator, true>(
            sum_workspace,
            sum_workspace,
            reduce_mask,
            m_actual,
            1,
            1,
            RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0);


        uint64_t add_final_mask = (m_actual < repeat_size) ? m_actual : repeat_size;
        uint64_t max_final_mask = (m_actual < repeat_size) ? m_actual : repeat_size;

        sum_params.dstRepStride = 8;
        sum_params.src0RepStride = 8;
        sum_params.src1RepStride = 8;

        max_params.dstRepStride = 8;
        max_params.src0RepStride = 8;
        max_params.src1RepStride = 8;

        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Add<ElementA, true>(
            dstTensorSum,
            sum_workspace,
            dstTensorSum,
            add_final_mask,
            CeilDiv(m_round, repeat_size),
            sum_params);

        AscendC::Max<ElementA, true>(
            dstTensorMax,
            srcTensor_m,
            dstTensorMax,
            max_final_mask,
            CeilDiv(m_round, repeat_size),
            max_params);
        
    }
};

template <>
struct TileFaultSum<Arch::AtlasA2,
                Gemv::helper::FT_REDUCE_TYPE::SUM_MAX_MIXED,
                Gemm::GemmType<half, layout::RowMajor>,
                Gemm::GemmType<half, layout::VectorLayout>,
                void>
{
    using ElementA = half;
    using ElementX = half;
    using ElementY = half;
    using ElementAccumulator = ElementY;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // Methods

    CATLASS_DEVICE
    TileFaultSum() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensorSum,
        AscendC::LocalTensor<ElementY> dstTensorMax,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        AscendC::LocalTensor<ElementA> sum_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = repeat_size;
        uint32_t repeat_num = n_actual / repeat_size;
        uint32_t remain = n_actual % repeat_size;

        // m_actual *
        uint64_t add_mask = repeat_size;
        uint64_t max_mask = repeat_size;

        /*
        控制操作数地址步长的参数。BinaryRepeatParams类型，
        包含操作数相邻迭代间相同datablock的地址步长，
        操作数同一迭代内不同datablock的地址步长等参数。

        相邻迭代间的地址步长参数说明请参考repeatStride；
        同一迭代内datablock的地址步长参数说明请参考dataBlockStride。
        */

        AscendC::BinaryRepeatParams max_params;
        max_params.dstBlkStride = 1;
        max_params.src0BlkStride = 1;
        max_params.src1BlkStride = 1;
        max_params.dstRepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        max_params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        max_params.src1RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

        AscendC::BinaryRepeatParams sum_params;
        sum_params.dstBlkStride = 1;
        sum_params.src0BlkStride = 1;
        sum_params.src1BlkStride = 1;

        // params.dstRepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

        sum_params.dstRepStride = RoundUp(repeat_size, repeat_size) / (BYTE_PER_C0 / sizeof(ElementAccumulator));
        sum_params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        sum_params.src1RepStride = RoundUp(repeat_size, repeat_size) / (BYTE_PER_C0 / sizeof(ElementAccumulator));


        AscendC::Duplicate<ElementAccumulator>(
                sum_workspace,
                (ElementAccumulator)0.0,
                repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
                CeilDiv(m_actual * repeat_size, repeat_size), // 求行和
                1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
                8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        );

        AscendC::PipeBarrier<PIPE_V>();
        
        AscendC::Add<ElementAccumulator, true>(
            sum_workspace,
            srcTensor_m,
            sum_workspace,
            add_mask,
            m_actual,
            sum_params);
        
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t i = 1; i < repeat_num; i++) {
            uint32_t offset = i * repeat_size;

            AscendC::Add<ElementAccumulator, true>(
                sum_workspace,
                srcTensor_m[offset],
                sum_workspace,
                add_mask,
                m_actual,
                sum_params);
            
            AscendC::Max<ElementA, true>(
                srcTensor_m,
                srcTensor_m[offset],
                srcTensor_m,
                max_mask,
                m_actual,
                max_params);
                
            AscendC::PipeBarrier<PIPE_V>();
        }

        if (remain > 0)
        {
            uint32_t offset = repeat_num * repeat_size;
            if (offset + remain > n_actual)
            {
                remain = n_actual - offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;
            AscendC::Add<ElementAccumulator, true>(
                sum_workspace,
                srcTensor_m[offset],
                sum_workspace,
                remain_mask,
                m_actual,
                sum_params);
            
            AscendC::Max<ElementA, true>(
                srcTensor_m,
                srcTensor_m[offset],
                srcTensor_m,
                remain_mask,
                m_actual,
                max_params);
        }
        
        uint64_t reduce_mask = (repeat_num == 0) ? remain : repeat_size;
        AscendC::PipeBarrier<PIPE_V>();
        
        AscendC::WholeReduceMax<ElementA, true>(
            srcTensor_m,
            srcTensor_m,
            reduce_mask,
            m_actual,
            1,
            1,
            RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE);

        AscendC::WholeReduceSum<ElementAccumulator, true>(
            sum_workspace,
            sum_workspace,
            reduce_mask,
            m_actual,
            1,
            1,
            RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0);


        uint64_t add_final_mask = (m_actual < repeat_size) ? m_actual : repeat_size;
        uint64_t max_final_mask = (m_actual < repeat_size) ? m_actual : repeat_size;

        sum_params.dstRepStride = 8;
        sum_params.src0RepStride = 8;
        sum_params.src1RepStride = 8;

        max_params.dstRepStride = 8;
        max_params.src0RepStride = 8;
        max_params.src1RepStride = 8;

        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Add<ElementA, true>(
            dstTensorSum,
            sum_workspace,
            dstTensorSum,
            add_final_mask,
            CeilDiv(m_round, repeat_size),
            sum_params);

        AscendC::Max<ElementA, true>(
            dstTensorMax,
            srcTensor_m,
            dstTensorMax,
            max_final_mask,
            CeilDiv(m_round, repeat_size),
            max_params);
        
    }
};


template <>
struct TileFaultSum<Arch::AtlasA2,
                Gemv::helper::FT_REDUCE_TYPE::SUM_MAX,
                Gemm::GemmType<half, layout::RowMajor>,
                Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementA = half;
    using ElementX = half;
    using ElementY = float;
    using ElementAccumulator = ElementY;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);
    static constexpr uint32_t DST_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);

    // Methods

    CATLASS_DEVICE
    TileFaultSum() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensorSum,
        AscendC::LocalTensor<ElementY> dstTensorMax,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        AscendC::LocalTensor<ElementA> sum_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);

        uint32_t dstOffset = m_round;

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t dst_repeat_size = DST_ELE_NUM_PER_C0 * 8;
        uint32_t mask = repeat_size;
        uint32_t repeat_num = n_actual / repeat_size;
        uint32_t remain = n_actual % repeat_size;

        // m_actual *
        uint64_t add_mask = repeat_size;
        uint64_t max_mask = repeat_size;

        /*
        控制操作数地址步长的参数。BinaryRepeatParams类型，
        包含操作数相邻迭代间相同datablock的地址步长，
        操作数同一迭代内不同datablock的地址步长等参数。

        相邻迭代间的地址步长参数说明请参考repeatStride；
        同一迭代内datablock的地址步长参数说明请参考dataBlockStride。
        */

        AscendC::BinaryRepeatParams max_params;
        max_params.dstBlkStride = 1;
        max_params.src0BlkStride = 1;
        max_params.src1BlkStride = 1;
        max_params.dstRepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        max_params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        max_params.src1RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

        AscendC::BinaryRepeatParams sum_params;
        sum_params.dstBlkStride = 1;
        sum_params.src0BlkStride = 1;
        sum_params.src1BlkStride = 1;

        // params.dstRepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

        sum_params.dstRepStride = RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0;
        sum_params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        sum_params.src1RepStride = RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0;

        

        // AscendC::Duplicate<ElementAccumulator>(
        //     temp,
        //     (ElementAccumulator)0.0,
        //     repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
        //     // 我们的目标是求行和，所以我们的输出空间是行的倍数，且是datablock的倍数
        //     // 因此在这里，我们在赋值0时通过每次内部迭代对应一个输出行的局部和存储与accum空间
        //     CeilDiv(m_actual * repeat_size, repeat_size), // 求行和
        //     1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
        //     8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        // ); // 总的来说，我们开辟
        // AscendC::PipeBarrier<PIPE_V>();
        
        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Add(const LocalTensor<T>& dstLocal, 
        const LocalTensor<T>& src0Local, 
        const LocalTensor<T>& src1Local, 
        uint64_t mask, 
        const uint8_t repeatTimes, 
        const BinaryRepeatParams& repeatParams)
        */


        AscendC::Duplicate<ElementA>(
                sum_workspace,
                (ElementA)0.0,
                repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
                CeilDiv(m_actual * repeat_size, repeat_size), // 求行和
                1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
                8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        );

        AscendC::PipeBarrier<PIPE_V>();
        
        AscendC::Add<ElementA, true>(
            sum_workspace,
            srcTensor_m,
            sum_workspace,
            add_mask,
            m_actual,
            sum_params);
        
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t i = 1; i < repeat_num; i++) {
            uint32_t offset = i * repeat_size;

            AscendC::Add<ElementA, true>(
                sum_workspace,
                srcTensor_m[offset],
                sum_workspace,
                add_mask,
                m_actual,
                sum_params);
            
            AscendC::Max<ElementA, true>(
                srcTensor_m,
                srcTensor_m[offset],
                srcTensor_m,
                max_mask,
                m_actual,
                max_params);
                
            AscendC::PipeBarrier<PIPE_V>();
        }

        if (remain > 0)
        {
            uint32_t offset = repeat_num * repeat_size;
            if (offset + remain > n_actual)
            {
                remain = n_actual - offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;
            AscendC::Add<ElementA, true>(
                sum_workspace,
                srcTensor_m[offset],
                sum_workspace,
                remain_mask,
                m_actual,
                sum_params);
            
            AscendC::Max<ElementA, true>(
                srcTensor_m,
                srcTensor_m[offset],
                srcTensor_m,
                remain_mask,
                m_actual,
                max_params);
        }
        
        uint64_t reduce_mask = (repeat_num == 0) ? remain : repeat_size;
        AscendC::PipeBarrier<PIPE_V>();
        
        AscendC::WholeReduceMax<ElementA, true>(
            srcTensor_m,
            srcTensor_m,
            reduce_mask,
            m_actual,
            1,
            1,
            RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE);

        AscendC::WholeReduceSum<ElementA, true>(
            sum_workspace,
            sum_workspace,
            reduce_mask,
            m_actual,
            1,
            1,
            RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0);

        AscendC::PipeBarrier<PIPE_V>();
        
        AscendC::Cast<ElementY, ElementA>(
            dstTensorMax[dstOffset],
            srcTensor_m,
            AscendC::RoundMode::CAST_NONE, m_actual);

        AscendC::Cast<ElementY, ElementA>(
            dstTensorSum[dstOffset],
            sum_workspace,
            AscendC::RoundMode::CAST_NONE, m_actual);
        
        AscendC::PipeBarrier<PIPE_V>();

        uint64_t add_final_mask = (m_actual < dst_repeat_size) ? m_actual : dst_repeat_size;
        uint64_t max_final_mask = (m_actual < dst_repeat_size) ? m_actual : dst_repeat_size;

        sum_params.dstRepStride = 8;
        sum_params.src0RepStride = 8;
        sum_params.src1RepStride = 8;

        max_params.dstRepStride = 8;
        max_params.src0RepStride = 8;
        max_params.src1RepStride = 8;

        AscendC::Add<ElementY, true>(
            dstTensorSum,
            dstTensorSum[dstOffset],
            dstTensorSum,
            add_final_mask,
            CeilDiv(m_round, dst_repeat_size),
            sum_params);

        AscendC::Max<ElementY, true>(
            dstTensorMax,
            dstTensorMax[dstOffset],
            dstTensorMax,
            max_final_mask,
            CeilDiv(m_round, dst_repeat_size),
            max_params);
        
    }
};


template <>
struct TileFaultSum<Arch::AtlasA2,
                Gemv::helper::FT_REDUCE_TYPE::SUM_MAX_MIXED,
                Gemm::GemmType<half, layout::RowMajor>,
                Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementA = half;
    using ElementX = half;
    using ElementY = float;
    using ElementAccumulator = ElementY;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);
    static constexpr uint32_t DST_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);

    // Methods

    CATLASS_DEVICE
    TileFaultSum() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensorSum,
        AscendC::LocalTensor<ElementA> dstTensorMax,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        AscendC::LocalTensor<ElementA> sum_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);

        uint32_t dstOffset = m_round;

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t dst_repeat_size = DST_ELE_NUM_PER_C0 * 8;
        uint32_t mask = repeat_size;
        uint32_t repeat_num = n_actual / repeat_size;
        uint32_t remain = n_actual % repeat_size;

        // m_actual *
        uint64_t add_mask = repeat_size;
        uint64_t max_mask = repeat_size;

        /*
        控制操作数地址步长的参数。BinaryRepeatParams类型，
        包含操作数相邻迭代间相同datablock的地址步长，
        操作数同一迭代内不同datablock的地址步长等参数。

        相邻迭代间的地址步长参数说明请参考repeatStride；
        同一迭代内datablock的地址步长参数说明请参考dataBlockStride。
        */

        AscendC::BinaryRepeatParams max_params;
        max_params.dstBlkStride = 1;
        max_params.src0BlkStride = 1;
        max_params.src1BlkStride = 1;
        max_params.dstRepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        max_params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        max_params.src1RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

        AscendC::BinaryRepeatParams sum_params;
        sum_params.dstBlkStride = 1;
        sum_params.src0BlkStride = 1;
        sum_params.src1BlkStride = 1;

        // params.dstRepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

        sum_params.dstRepStride = RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0;
        sum_params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        sum_params.src1RepStride = RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0;
        
        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Add(const LocalTensor<T>& dstLocal, 
        const LocalTensor<T>& src0Local, 
        const LocalTensor<T>& src1Local, 
        uint64_t mask, 
        const uint8_t repeatTimes, 
        const BinaryRepeatParams& repeatParams)
        */


        AscendC::Duplicate<ElementA>(
                sum_workspace,
                (ElementA)0.0,
                repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
                CeilDiv(m_actual * repeat_size, repeat_size), // 求行和
                1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
                8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        );

        AscendC::PipeBarrier<PIPE_V>();
        
        AscendC::Add<ElementA, true>(
            sum_workspace,
            srcTensor_m,
            sum_workspace,
            add_mask,
            m_actual,
            sum_params);
        
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t i = 1; i < repeat_num; i++) {
            uint32_t offset = i * repeat_size;

            AscendC::Add<ElementA, true>(
                sum_workspace,
                srcTensor_m[offset],
                sum_workspace,
                add_mask,
                m_actual,
                sum_params);
            
            AscendC::Max<ElementA, true>(
                srcTensor_m,
                srcTensor_m[offset],
                srcTensor_m,
                max_mask,
                m_actual,
                max_params);
                
            AscendC::PipeBarrier<PIPE_V>();
        }

        if (remain > 0)
        {
            uint32_t offset = repeat_num * repeat_size;
            if (offset + remain > n_actual)
            {
                remain = n_actual - offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;
            AscendC::Add<ElementA, true>(
                sum_workspace,
                srcTensor_m[offset],
                sum_workspace,
                remain_mask,
                m_actual,
                sum_params);
            
            AscendC::Max<ElementA, true>(
                srcTensor_m,
                srcTensor_m[offset],
                srcTensor_m,
                remain_mask,
                m_actual,
                max_params);
        }
        
        uint64_t reduce_mask = (repeat_num == 0) ? remain : repeat_size;
        AscendC::PipeBarrier<PIPE_V>();
        
        AscendC::WholeReduceMax<ElementA, true>(
            srcTensor_m,
            srcTensor_m,
            reduce_mask,
            m_actual,
            1,
            1,
            RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE);

        AscendC::WholeReduceSum<ElementA, true>(
            sum_workspace,
            sum_workspace,
            reduce_mask,
            m_actual,
            1,
            1,
            RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0);

        AscendC::PipeBarrier<PIPE_V>();
        
        // AscendC::Cast<ElementY, ElementA>(
        //     dstTensorMax[dstOffset],
        //     srcTensor_m,
        //     AscendC::RoundMode::CAST_NONE, m_actual);

        AscendC::Cast<ElementY, ElementA>(
            dstTensorSum[dstOffset],
            sum_workspace,
            AscendC::RoundMode::CAST_NONE, m_actual);
        
        AscendC::PipeBarrier<PIPE_V>();

        uint64_t add_final_mask = (m_actual < dst_repeat_size) ? m_actual : dst_repeat_size;
        uint64_t max_final_mask = (m_actual < repeat_size) ? m_actual : repeat_size;

        sum_params.dstRepStride = 8;
        sum_params.src0RepStride = 8;
        sum_params.src1RepStride = 8;

        max_params.dstRepStride = 8;
        max_params.src0RepStride = 8;
        max_params.src1RepStride = 8;

        AscendC::Add<ElementY, true>(
            dstTensorSum,
            dstTensorSum[dstOffset],
            dstTensorSum,
            add_final_mask,
            CeilDiv(m_round, dst_repeat_size),
            sum_params);

        AscendC::Max<ElementA, true>(
            dstTensorMax,
            srcTensor_m,
            dstTensorMax,
            max_final_mask,
            CeilDiv(m_round, repeat_size),
            max_params);
        
    }
};

template <>
struct TileFaultSum<Arch::AtlasA2,
                Gemv::helper::FT_REDUCE_TYPE::SUM,
                Gemm::GemmType<half, layout::RowMajor>,
                Gemm::GemmType<half, layout::VectorLayout>,
                void>
{
    using ElementA = half;
    using ElementX = half;
    using ElementY = half;
    using ElementAccumulator = ElementY;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // Mehtods

    CATLASS_DEVICE
    TileFaultSum() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        AscendC::LocalTensor<ElementA> temp,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = repeat_size;
        uint32_t repeat_num = n_actual / repeat_size;
        uint32_t remain = n_actual % repeat_size;

        // m_actual * 
        uint64_t add_mask = repeat_size;
        AscendC::BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;
        params.dstRepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        params.src1RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        for (uint32_t i = 1; i < repeat_num; i++) {
            uint32_t offset = i * repeat_size;
            AscendC::Add<ElementA, true>(
                srcTensor_m,
                srcTensor_m[offset],
                srcTensor_m,
                add_mask,
                m_actual,
                params);
            AscendC::PipeBarrier<PIPE_V>();
        }

        if (remain > 0)
        {
            uint32_t offset = repeat_num * repeat_size;
            if (offset + remain > n_actual)
            {
                remain = n_actual - offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;
            AscendC::Add<ElementA, true>(
                srcTensor_m,
                srcTensor_m[offset],
                srcTensor_m,
                remain_mask,
                m_actual,
                params);
        }

        uint64_t reduce_mask = (repeat_num == 0) ? remain : repeat_size;
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum<ElementA, true>(
            srcTensor_m,
            srcTensor_m,
            reduce_mask,
            m_actual,
            1,
            1,
            RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0);

        add_mask = (m_actual < repeat_size) ? m_actual : repeat_size;
        params.dstRepStride = 8;
        params.src0RepStride = 8;
        params.src1RepStride = 8;

        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<ElementA, true>(
            dstTensor,
            srcTensor_m,
            dstTensor,
            add_mask,
            CeilDiv(m_round, repeat_size),
            params);
    }
};

template <>
struct TileFaultSum<Arch::AtlasA2,
                Gemv::helper::FT_REDUCE_TYPE::SUM,
                Gemm::GemmType<half, layout::RowMajor>,
                Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementA = half;
    using ElementX = half;
    using ElementY = float;
    using ElementAccumulator = ElementY;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);
    static constexpr uint32_t DST_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);

    // Mehtods

    CATLASS_DEVICE
    TileFaultSum() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        AscendC::LocalTensor<ElementA> temp,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t dst_repeat_size = DST_ELE_NUM_PER_C0 * 8;
        uint32_t mask = repeat_size;
        uint32_t repeat_num = n_actual / repeat_size;
        uint32_t remain = n_actual % repeat_size;

        // m_actual * 
        uint64_t add_mask = repeat_size;
        AscendC::BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;
        params.dstRepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        params.src1RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        for (uint32_t i = 1; i < repeat_num; i++) {
            uint32_t offset = i * repeat_size;
            AscendC::Add<ElementA, true>(
                srcTensor_m,
                srcTensor_m[offset],
                srcTensor_m,
                add_mask,
                m_actual,
                params);
            AscendC::PipeBarrier<PIPE_V>();
        }

        if (remain > 0)
        {
            uint32_t offset = repeat_num * repeat_size;
            if (offset + remain > n_actual)
            {
                remain = n_actual - offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;
            AscendC::Add<ElementA, true>(
                srcTensor_m,
                srcTensor_m[offset],
                srcTensor_m,
                remain_mask,
                m_actual,
                params);
        }

        uint64_t reduce_mask = (repeat_num == 0) ? remain : repeat_size;
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum<ElementA, true>(
            srcTensor_m,
            srcTensor_m,
            reduce_mask,
            m_actual,
            1,
            1,
            RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0);

        AscendC::PipeBarrier<PIPE_V>();

        add_mask = (m_actual < dst_repeat_size) ? m_actual : dst_repeat_size;

        uint32_t dstOffset = m_round;

        AscendC::Cast<ElementY, ElementA>(
            dstTensor[dstOffset],
            srcTensor_m,
            AscendC::RoundMode::CAST_NONE, m_actual);
        AscendC::PipeBarrier<PIPE_V>();

        params.dstRepStride = 8;
        params.src0RepStride = 8;
        params.src1RepStride = 8;

        
        AscendC::Add<ElementY, true>(
            dstTensor,
            dstTensor[dstOffset],
            dstTensor,
            add_mask,
            CeilDiv(m_round, dst_repeat_size),
            params);
    }
};


template <
    class ElementA,
    class ElementY
>
struct TileFaultSum<Arch::AtlasA2,
                Gemv::helper::FT_REDUCE_TYPE::SUM,
                Gemm::GemmType<ElementA, layout::ColumnMajor>,
                Gemm::GemmType<ElementY, layout::VectorLayout>,
                void>
{
    // using ElementAccumulator = ElementY;
    // using LayoutDst = layout::ColumnMajor;
    // using LayoutSrc = layout::ColumnMajor;
    // static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // // Mehtods

    // CATLASS_DEVICE
    // TileFaultSum() {};

    // CATLASS_DEVICE
    // void operator()(
    //     AscendC::LocalTensor<ElementY> dstTensor,
    //     AscendC::LocalTensor<ElementX> srcTensor_v,
    //     AscendC::LocalTensor<ElementA> srcTensor_m,
    //     AscendC::LocalTensor<ElementAccumulator> temp,
    //     LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    // )
    // {
    //     uint32_t m_actual = layoutSrc.shape(0);
    //     uint32_t n_actual = layoutSrc.shape(1);
    //     uint32_t m_round = layoutDst.shape(0);
    //     uint32_t n_round = layoutDst.shape(1);
    //     AscendC::SetMaskCount();
    //     AscendC::SetVectorMask<ElementAccumulator, AscendC::MaskMode::COUNTER>(m_actual);
    //     AscendC::Duplicate<ElementAccumulator, false>(
    //         temp,
    //         (ElementAccumulator)0.0,
    //         AscendC::MASK_PLACEHOLDER,
    //         1,
    //         1,
    //         8);
    //     AscendC::PipeBarrier<PIPE_V>();

    //     AscendC::SetFlag<AscendC::HardEvent::V_S>((event_t)(0));
    //     AscendC::WaitFlag<AscendC::HardEvent::V_S>((event_t)(0));

    //     AscendC::UnaryRepeatParams params;
    //     params.dstBlkStride = 1;
    //     params.srcBlkStride = 1;
    //     params.dstRepStride = 8;
    //     params.srcRepStride = 4;
    //     for (uint32_t i = 0; i < n_actual; i++)
    //     {
    //         AscendC::Axpy<ElementAccumulator, ElementA, false>(
    //             temp,
    //             srcTensor_m[i * m_round],
    //             srcTensor_v.GetValue(i),
    //             AscendC::MASK_PLACEHOLDER,
    //             1,
    //             params);
    //         AscendC::PipeBarrier<PIPE_V>();
    //     }
    //     params.dstRepStride = 4;
    //     params.srcRepStride = 8;
    //     AscendC::Cast<ElementA, ElementAccumulator, false>(
    //         srcTensor_m,
    //         temp,
    //         AscendC::RoundMode::CAST_NONE,
    //         AscendC::MASK_PLACEHOLDER,
    //         1,
    //         params);
    //     AscendC::BinaryRepeatParams addparams;
    //     addparams.dstBlkStride = 1;
    //     addparams.src0BlkStride = 1;
    //     addparams.src1BlkStride = 1;
    //     addparams.dstRepStride = 8;
    //     addparams.src0RepStride = 8;
    //     addparams.src1RepStride = 8;
    //     AscendC::PipeBarrier<PIPE_V>();
    //     AscendC::Add<ElementA, false>(
    //         dstTensor,
    //         srcTensor_m,
    //         dstTensor,
    //         AscendC::MASK_PLACEHOLDER,
    //         1,
    //         addparams);
    //     AscendC::SetMaskNorm();
    //     AscendC::ResetMask();
    // }
};

template <>
struct TileFaultSum<Arch::AtlasA2,
                Gemv::helper::FT_REDUCE_TYPE::SUM,
                Gemm::GemmType<float, layout::ColumnMajor>,
                Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementA = float;
    using ElementY = float;
    using LayoutDst = layout::ColumnMajor;
    using LayoutSrc = layout::ColumnMajor;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // Mehtods

    CATLASS_DEVICE
    TileFaultSum() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        AscendC::LocalTensor<ElementY> temp,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);
        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementY);
        uint32_t mask = temp_repeat_size;
        uint32_t repeattimes = CeilDiv(m_actual, temp_repeat_size);

        uint64_t add_mask = (m_actual < temp_repeat_size) ? m_actual : temp_repeat_size;
        AscendC::BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;
        params.dstRepStride = 8;
        params.src0RepStride = 8;
        params.src1RepStride = 8;
        for (uint32_t i = 0; i < n_actual; i++) {
            AscendC::Add<ElementA, true>(
                dstTensor,
                srcTensor_m[i * m_round],
                dstTensor,
                (uint64_t)add_mask,
                CeilDiv(m_round, temp_repeat_size),
                params);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
};

template <>
struct TileFaultSum<Arch::AtlasA2,
                Gemv::helper::FT_REDUCE_TYPE::SUM,
                Gemm::GemmType<half, layout::ColumnMajor>,
                Gemm::GemmType<half, layout::VectorLayout>,
                void>
{
    using ElementA = half;
    using ElementY = half;
    using LayoutDst = layout::ColumnMajor;
    using LayoutSrc = layout::ColumnMajor;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // Mehtods

    CATLASS_DEVICE
    TileFaultSum() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        AscendC::LocalTensor<ElementY> temp,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);
        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementY);
        uint32_t mask = temp_repeat_size;
        uint32_t repeattimes = CeilDiv(m_actual, temp_repeat_size);

        uint64_t add_mask = (m_actual < temp_repeat_size) ? m_actual : temp_repeat_size;
        AscendC::BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;
        params.dstRepStride = 8;
        params.src0RepStride = 8;
        params.src1RepStride = 8;
        for (uint32_t i = 0; i < n_actual; i++) {
            AscendC::Add<ElementA, true>(
                dstTensor,
                srcTensor_m[i * m_round],
                dstTensor,
                (uint64_t)add_mask,
                CeilDiv(m_round, temp_repeat_size),
                params);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
};

template <>
struct TileFaultSum<Arch::AtlasA2,
                Gemv::helper::FT_REDUCE_TYPE::MAX_MIN,
                Gemm::GemmType<float, layout::RowMajor>,
                Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementA = float;
    using ElementX = float;
    using ElementY = float;
    using ElementAccumulator = ElementY;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // Methods

    CATLASS_DEVICE
    TileFaultSum() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensorMin,
        AscendC::LocalTensor<ElementY> dstTensorMax,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        AscendC::LocalTensor<ElementA> min_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = repeat_size;
        uint32_t repeat_num = n_actual / repeat_size;
        uint32_t remain = n_actual % repeat_size;

        // m_actual *
        uint64_t min_mask = repeat_size;
        uint64_t max_mask = repeat_size;

        /*
        控制操作数地址步长的参数。BinaryRepeatParams类型，
        包含操作数相邻迭代间相同datablock的地址步长，
        操作数同一迭代内不同datablock的地址步长等参数。

        相邻迭代间的地址步长参数说明请参考repeatStride；
        同一迭代内datablock的地址步长参数说明请参考dataBlockStride。
        */

        AscendC::BinaryRepeatParams max_params;
        max_params.dstBlkStride = 1;
        max_params.src0BlkStride = 1;
        max_params.src1BlkStride = 1;
        max_params.dstRepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        max_params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        max_params.src1RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

        AscendC::BinaryRepeatParams min_params;
        min_params.dstBlkStride = 1;
        min_params.src0BlkStride = 1;
        min_params.src1BlkStride = 1;

        // params.dstRepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

        min_params.dstRepStride = RoundUp(repeat_size, repeat_size) / (BYTE_PER_C0 / sizeof(ElementAccumulator));
        min_params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        min_params.src1RepStride = RoundUp(repeat_size, repeat_size) / (BYTE_PER_C0 / sizeof(ElementAccumulator));

        

        // AscendC::Duplicate<ElementAccumulator>(
        //     temp,
        //     (ElementAccumulator)0.0,
        //     repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
        //     // 我们的目标是求行和，所以我们的输出空间是行的倍数，且是datablock的倍数
        //     // 因此在这里，我们在赋值0时通过每次内部迭代对应一个输出行的局部和存储与accum空间
        //     CeilDiv(m_actual * repeat_size, repeat_size), // 求行和
        //     1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
        //     8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        // ); // 总的来说，我们开辟
        // AscendC::PipeBarrier<PIPE_V>();
        
        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Add(const LocalTensor<T>& dstLocal, 
        const LocalTensor<T>& src0Local, 
        const LocalTensor<T>& src1Local, 
        uint64_t mask, 
        const uint8_t repeatTimes, 
        const BinaryRepeatParams& repeatParams)
        */


        AscendC::Duplicate<ElementA>(
                min_workspace,
                (ElementA)0.0,
                repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
                CeilDiv(m_actual * repeat_size, repeat_size), // 求行最小值
                1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
                8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        );

        AscendC::PipeBarrier<PIPE_V>();
        
        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Min(
            const LocalTensor<T>& dstLocal, 
            const LocalTensor<T>& src0Local, 
            const LocalTensor<T>& src1Local, uint64_t mask, 
            const uint8_t repeatTimes, 
            const BinaryRepeatParams& repeatParams)
        */
        AscendC::Min<ElementA, true>(
            min_workspace,
            srcTensor_m,
            min_workspace,
            min_mask,
            m_actual,
            min_params);
        
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t i = 1; i < repeat_num; i++) {
            uint32_t offset = i * repeat_size;

            AscendC::Min<ElementA, true>(
                min_workspace,
                srcTensor_m[offset],
                min_workspace,
                min_mask,
                m_actual,
                min_params);
            
            AscendC::Max<ElementA, true>(
                srcTensor_m,
                srcTensor_m[offset],
                srcTensor_m,
                max_mask,
                m_actual,
                max_params);
                
            AscendC::PipeBarrier<PIPE_V>();
        }

        if (remain > 0)
        {
            uint32_t offset = repeat_num * repeat_size;
            if (offset + remain > n_actual)
            {
                remain = n_actual - offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;
            AscendC::Min<ElementA, true>(
                min_workspace,
                srcTensor_m[offset],
                min_workspace,
                remain_mask,
                m_actual,
                min_params);
            
            AscendC::Max<ElementA, true>(
                srcTensor_m,
                srcTensor_m[offset],
                srcTensor_m,
                remain_mask,
                m_actual,
                max_params);
        }
        
        uint64_t reduce_mask = (repeat_num == 0) ? remain : repeat_size;
        AscendC::PipeBarrier<PIPE_V>();
        
        AscendC::WholeReduceMax<ElementA, true>(
            srcTensor_m,
            srcTensor_m,
            reduce_mask,
            m_actual,
            1,
            1,
            RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE);

        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void WholeReduceMin(const LocalTensor<T>& dstLocal, 
            const LocalTensor<T>& srcLocal, 
            const int32_t mask, const int32_t repeatTimes,
            const int32_t dstRepStride, const int32_t srcBlkStride,
            const int32_t srcRepStride,
            ReduceOrder order = ReduceOrder::ORDER_VALUE_INDEX)

        ORDER_VALUE_INDEX：表示value位于低半部，返回结果存储顺序为[value, index]。
        ORDER_INDEX_VALUE：表示index位于低半部，返回结果存储顺序为[index, value]。
        ORDER_ONLY_VALUE：表示只返回最值，返回结果存储顺序为[value]。
        ORDER_ONLY_INDEX：表示只返回最值索引，返回结果存储顺序为[index]。
        */

        AscendC::WholeReduceMin<ElementA, true>(
            min_workspace,
            min_workspace,
            reduce_mask,
            m_actual,
            1,
            1,
            RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE
        );

        uint64_t min_final_mask = (m_actual < repeat_size) ? m_actual : repeat_size;
        uint64_t max_final_mask = (m_actual < repeat_size) ? m_actual : repeat_size;

        min_params.dstRepStride = 8;
        min_params.src0RepStride = 8;
        min_params.src1RepStride = 8;

        max_params.dstRepStride = 8;
        max_params.src0RepStride = 8;
        max_params.src1RepStride = 8;

        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Min<ElementA, true>(
            dstTensorMin,
            min_workspace,
            dstTensorMin,
            min_final_mask,
            CeilDiv(m_round, repeat_size),
            min_params);

        AscendC::Max<ElementA, true>(
            dstTensorMax,
            srcTensor_m,
            dstTensorMax,
            max_final_mask,
            CeilDiv(m_round, repeat_size),
            max_params);   
    }
};


template <>
struct TileFaultSum<Arch::AtlasA2,
                Gemv::helper::FT_REDUCE_TYPE::MAX_MIN,
                Gemm::GemmType<half, layout::RowMajor>,
                Gemm::GemmType<half, layout::VectorLayout>,
                void>
{
    using ElementA = half;
    using ElementX = half;
    using ElementY = half;
    using ElementAccumulator = ElementY;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // Methods

    CATLASS_DEVICE
    TileFaultSum() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensorMin,
        AscendC::LocalTensor<ElementY> dstTensorMax,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        AscendC::LocalTensor<ElementA> min_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = repeat_size;
        uint32_t repeat_num = n_actual / repeat_size;
        uint32_t remain = n_actual % repeat_size;

        // m_actual *
        uint64_t min_mask = repeat_size;
        uint64_t max_mask = repeat_size;

        /*
        控制操作数地址步长的参数。BinaryRepeatParams类型，
        包含操作数相邻迭代间相同datablock的地址步长，
        操作数同一迭代内不同datablock的地址步长等参数。

        相邻迭代间的地址步长参数说明请参考repeatStride；
        同一迭代内datablock的地址步长参数说明请参考dataBlockStride。
        */

        AscendC::BinaryRepeatParams max_params;
        max_params.dstBlkStride = 1;
        max_params.src0BlkStride = 1;
        max_params.src1BlkStride = 1;
        max_params.dstRepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        max_params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        max_params.src1RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

        AscendC::BinaryRepeatParams min_params;
        min_params.dstBlkStride = 1;
        min_params.src0BlkStride = 1;
        min_params.src1BlkStride = 1;

        // params.dstRepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

        min_params.dstRepStride = RoundUp(repeat_size, repeat_size) / (BYTE_PER_C0 / sizeof(ElementAccumulator));
        min_params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        min_params.src1RepStride = RoundUp(repeat_size, repeat_size) / (BYTE_PER_C0 / sizeof(ElementAccumulator));

        

        // AscendC::Duplicate<ElementAccumulator>(
        //     temp,
        //     (ElementAccumulator)0.0,
        //     repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
        //     // 我们的目标是求行和，所以我们的输出空间是行的倍数，且是datablock的倍数
        //     // 因此在这里，我们在赋值0时通过每次内部迭代对应一个输出行的局部和存储与accum空间
        //     CeilDiv(m_actual * repeat_size, repeat_size), // 求行和
        //     1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
        //     8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        // ); // 总的来说，我们开辟
        // AscendC::PipeBarrier<PIPE_V>();
        
        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Add(const LocalTensor<T>& dstLocal, 
        const LocalTensor<T>& src0Local, 
        const LocalTensor<T>& src1Local, 
        uint64_t mask, 
        const uint8_t repeatTimes, 
        const BinaryRepeatParams& repeatParams)
        */


        AscendC::Duplicate<ElementA>(
                min_workspace,
                (ElementA)0.0,
                repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
                CeilDiv(m_actual * repeat_size, repeat_size), // 求行和
                1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
                8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        );

        AscendC::PipeBarrier<PIPE_V>();
        
        AscendC::Min<ElementA, true>(
            min_workspace,
            srcTensor_m,
            min_workspace,
            min_mask,
            m_actual,
            min_params);
        
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t i = 1; i < repeat_num; i++) {
            uint32_t offset = i * repeat_size;

            AscendC::Min<ElementA, true>(
                min_workspace,
                srcTensor_m[offset],
                min_workspace,
                min_mask,
                m_actual,
                min_params);
            
            AscendC::Max<ElementA, true>(
                srcTensor_m,
                srcTensor_m[offset],
                srcTensor_m,
                max_mask,
                m_actual,
                max_params);
                
            AscendC::PipeBarrier<PIPE_V>();
        }

        if (remain > 0)
        {
            uint32_t offset = repeat_num * repeat_size;
            if (offset + remain > n_actual)
            {
                remain = n_actual - offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;
            AscendC::Min<ElementA, true>(
                min_workspace,
                srcTensor_m[offset],
                min_workspace,
                remain_mask,
                m_actual,
                min_params);
            
            AscendC::Max<ElementA, true>(
                srcTensor_m,
                srcTensor_m[offset],
                srcTensor_m,
                remain_mask,
                m_actual,
                max_params);
        }
        
        uint64_t reduce_mask = (repeat_num == 0) ? remain : repeat_size;
        AscendC::PipeBarrier<PIPE_V>();
        
        AscendC::WholeReduceMax<ElementA, true>(
            srcTensor_m,
            srcTensor_m,
            reduce_mask,
            m_actual,
            1,
            1,
            RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE);

        AscendC::WholeReduceMin<ElementA, true>(
            min_workspace,
            min_workspace,
            reduce_mask,
            m_actual,
            1,
            1,
            RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE
        );


        uint64_t min_final_mask = (m_actual < repeat_size) ? m_actual : repeat_size;
        uint64_t max_final_mask = (m_actual < repeat_size) ? m_actual : repeat_size;

        min_params.dstRepStride = 8;
        min_params.src0RepStride = 8;
        min_params.src1RepStride = 8;

        max_params.dstRepStride = 8;
        max_params.src0RepStride = 8;
        max_params.src1RepStride = 8;

        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Min<ElementA, true>(
            dstTensorMin,
            min_workspace,
            dstTensorMin,
            min_final_mask,
            CeilDiv(m_round, repeat_size),
            min_params);

        AscendC::Max<ElementA, true>(
            dstTensorMax,
            srcTensor_m,
            dstTensorMax,
            max_final_mask,
            CeilDiv(m_round, repeat_size),
            max_params);
        
    }
};

template <>
struct TileFaultSum<Arch::AtlasA2,
                Gemv::helper::FT_REDUCE_TYPE::MAX_MIN,
                Gemm::GemmType<half, layout::RowMajor>,
                Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementA = half;
    using ElementX = half;
    using ElementY = float;
    using ElementAccumulator = ElementY;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);
    static constexpr uint32_t DST_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);

    // Methods

    CATLASS_DEVICE
    TileFaultSum() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensorMin,
        AscendC::LocalTensor<ElementY> dstTensorMax,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        AscendC::LocalTensor<ElementA> min_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);

        uint32_t dstOffset = m_round;

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t dst_repeat_size = DST_ELE_NUM_PER_C0 * 8;
        uint32_t mask = repeat_size;
        uint32_t repeat_num = n_actual / repeat_size;
        uint32_t remain = n_actual % repeat_size;

        // m_actual *
        uint64_t min_mask = repeat_size;
        uint64_t max_mask = repeat_size;

        /*
        控制操作数地址步长的参数。BinaryRepeatParams类型，
        包含操作数相邻迭代间相同datablock的地址步长，
        操作数同一迭代内不同datablock的地址步长等参数。

        相邻迭代间的地址步长参数说明请参考repeatStride；
        同一迭代内datablock的地址步长参数说明请参考dataBlockStride。
        */

        AscendC::BinaryRepeatParams max_params;
        max_params.dstBlkStride = 1;
        max_params.src0BlkStride = 1;
        max_params.src1BlkStride = 1;
        max_params.dstRepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        max_params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        max_params.src1RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

        AscendC::BinaryRepeatParams min_params;
        min_params.dstBlkStride = 1;
        min_params.src0BlkStride = 1;
        min_params.src1BlkStride = 1;

        // params.dstRepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

        min_params.dstRepStride = RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0;
        min_params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        min_params.src1RepStride = RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0;

        

        // AscendC::Duplicate<ElementAccumulator>(
        //     temp,
        //     (ElementAccumulator)0.0,
        //     repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
        //     // 我们的目标是求行和，所以我们的输出空间是行的倍数，且是datablock的倍数
        //     // 因此在这里，我们在赋值0时通过每次内部迭代对应一个输出行的局部和存储与accum空间
        //     CeilDiv(m_actual * repeat_size, repeat_size), // 求行和
        //     1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
        //     8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        // ); // 总的来说，我们开辟
        // AscendC::PipeBarrier<PIPE_V>();
        
        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Add(const LocalTensor<T>& dstLocal, 
        const LocalTensor<T>& src0Local, 
        const LocalTensor<T>& src1Local, 
        uint64_t mask, 
        const uint8_t repeatTimes, 
        const BinaryRepeatParams& repeatParams)
        */


        AscendC::Duplicate<ElementA>(
                min_workspace,
                (ElementA)0.0,
                repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
                CeilDiv(m_actual * repeat_size, repeat_size), // 求行和
                1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
                8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        );

        AscendC::PipeBarrier<PIPE_V>();
        
        AscendC::Min<ElementA, true>(
            min_workspace,
            srcTensor_m,
            min_workspace,
            min_mask,
            m_actual,
            min_params);
        
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t i = 1; i < repeat_num; i++) {
            uint32_t offset = i * repeat_size;

            AscendC::Min<ElementA, true>(
                min_workspace,
                srcTensor_m[offset],
                min_workspace,
                min_mask,
                m_actual,
                min_params);
            
            AscendC::Max<ElementA, true>(
                srcTensor_m,
                srcTensor_m[offset],
                srcTensor_m,
                max_mask,
                m_actual,
                max_params);
                
            AscendC::PipeBarrier<PIPE_V>();
        }

        if (remain > 0)
        {
            uint32_t offset = repeat_num * repeat_size;
            if (offset + remain > n_actual)
            {
                remain = n_actual - offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;
            AscendC::Min<ElementA, true>(
                min_workspace,
                srcTensor_m[offset],
                min_workspace,
                remain_mask,
                m_actual,
                min_params);
            
            AscendC::Max<ElementA, true>(
                srcTensor_m,
                srcTensor_m[offset],
                srcTensor_m,
                remain_mask,
                m_actual,
                max_params);
        }
        
        uint64_t reduce_mask = (repeat_num == 0) ? remain : repeat_size;
        AscendC::PipeBarrier<PIPE_V>();
        
        AscendC::WholeReduceMax<ElementA, true>(
            srcTensor_m,
            srcTensor_m,
            reduce_mask,
            m_actual,
            1,
            1,
            RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        
        AscendC::WholeReduceMin<ElementA, true>(
            min_workspace,
            min_workspace,
            reduce_mask,
            m_actual,
            1,
            1,
            RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE);

        AscendC::PipeBarrier<PIPE_V>();
        
        AscendC::Cast<ElementY, ElementA>(
            dstTensorMax[dstOffset],
            srcTensor_m,
            AscendC::RoundMode::CAST_NONE, m_actual);

        AscendC::Cast<ElementY, ElementA>(
            dstTensorMin[dstOffset],
            min_workspace,
            AscendC::RoundMode::CAST_NONE, m_actual);
        
        AscendC::PipeBarrier<PIPE_V>();

        uint64_t min_final_mask = (m_actual < dst_repeat_size) ? m_actual : dst_repeat_size;
        uint64_t max_final_mask = (m_actual < dst_repeat_size) ? m_actual : dst_repeat_size;

        min_params.dstRepStride = 8;
        min_params.src0RepStride = 8;
        min_params.src1RepStride = 8;

        max_params.dstRepStride = 8;
        max_params.src0RepStride = 8;
        max_params.src1RepStride = 8;

        AscendC::Min<ElementY, true>(
            dstTensorMin,
            dstTensorMin[dstOffset],
            dstTensorMin,
            min_final_mask,
            CeilDiv(m_round, dst_repeat_size),
            min_params);

        AscendC::Max<ElementY, true>(
            dstTensorMax,
            dstTensorMax[dstOffset],
            dstTensorMax,
            max_final_mask,
            CeilDiv(m_round, dst_repeat_size),
            max_params);
        
    }
};

template <>
struct TileFaultSumVmad<Arch::AtlasA2,
                Gemv::helper::FT_REDUCE_TYPE::SUM_VMAD,
                Gemm::GemmType<float, layout::RowMajor>,
                Gemm::GemmType<float, layout::VectorLayout>,
                Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementA = float;
    using ElementX = float;
    using ElementY = float;
    using ElementAccumulator = ElementY;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // Methods

    CATLASS_DEVICE
    TileFaultSumVmad() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensorVMAD,
        AscendC::LocalTensor<ElementY> dstTensorSUM,
        AscendC::LocalTensor<ElementX> srcTensor_v,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        AscendC::LocalTensor<ElementAccumulator> vmad_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementAccumulator);

        uint32_t vmad_mask = temp_repeat_size;
        uint32_t repeat_num = n_actual / temp_repeat_size;
        uint32_t remain = n_actual % temp_repeat_size;

        // m_actual *
        uint64_t add_mask = temp_repeat_size;

        /*
        控制操作数地址步长的参数。BinaryRepeatParams类型，
        包含操作数相邻迭代间相同datablock的地址步长，
        操作数同一迭代内不同datablock的地址步长等参数。

        相邻迭代间的地址步长参数说明请参考repeatStride；
        同一迭代内datablock的地址步长参数说明请参考dataBlockStride。
        */

        AscendC::BinaryRepeatParams add_params;
        add_params.dstBlkStride = 1;
        add_params.src0BlkStride = 1;
        add_params.src1BlkStride = 1;
        add_params.dstRepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        add_params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        add_params.src1RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

        // 将输出向量赋值为0
        AscendC::Duplicate<ElementAccumulator>(
            vmad_workspace,
            (ElementAccumulator)0.0,
            // 这是为了后续我们每次求全部行的局部行和，
            // 我们对每行局部和的存储空间设定为256byte对齐，
            // 这是因为每次赋值操作的 repeat 迭代内的运算能力是256byte
            temp_repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
            // 我们的目标是求行和，所以我们的输出空间是行的倍数，且是datablock的倍数
            // 因此在这里，我们在赋值0时通过每次内部迭代对应一个输出行的局部和存储与accum空间
            CeilDiv(m_round * temp_repeat_size, temp_repeat_size), // 求行和
            1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
            8 // 相邻赋值迭代间，矢量目的操作数不同 datablock chunk 间地址步长。
        ); // 总的来说，我们开辟了256 byte x m_round的空间，全部赋值为0，每256byte对应一个行的局部和存储与accum

        AscendC::PipeBarrier<PIPE_V>();

        uint32_t vmad_offset_raw = 0 * temp_repeat_size;
        AscendC::BinaryRepeatParams vmad_params;
        vmad_params.dstBlkStride = 1;
        vmad_params.src0BlkStride = 1;
        vmad_params.src1BlkStride = 1;

        vmad_params.dstRepStride = RoundUp(temp_repeat_size, temp_repeat_size) / (BYTE_PER_C0 / sizeof(ElementAccumulator));
        vmad_params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        vmad_params.src1RepStride = 0;

        if(repeat_num > 0){
            AscendC::MulAddDst<ElementAccumulator, ElementA, true>(
            vmad_workspace,
            srcTensor_m[vmad_offset_raw], // 这里是起始地址，每次外层迭代都是从第一行的相应列偏移开始的
            srcTensor_v[vmad_offset_raw],
            vmad_mask,
            m_actual,
            vmad_params);
        }else{
            // 如果没有完整的repeat_size，那么就直接对remain部分进行vmad
            uint64_t remain_mask = remain;
            AscendC::MulAddDst<ElementAccumulator, ElementA, true>(
                vmad_workspace,
                srcTensor_m[vmad_offset_raw], // 这里是起始地址，每次外层迭代都是从第一行的相应列偏移开始的
                srcTensor_v[vmad_offset_raw],
                remain_mask,
                m_actual,
                vmad_params);
        }

        AscendC::PipeBarrier<PIPE_V>();

        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Add(const LocalTensor<T>& dstLocal, 
        const LocalTensor<T>& src0Local, 
        const LocalTensor<T>& src1Local, 
        uint64_t mask, 
        const uint8_t repeatTimes, 
        const BinaryRepeatParams& repeatParams)
        */

        for (uint32_t i = 1; i < repeat_num; i++) {

            uint32_t offset = i * temp_repeat_size;

            AscendC::MulAddDst<ElementAccumulator, ElementA, true>(
                vmad_workspace,
                srcTensor_m[offset], // 这里是起始地址，每次外层迭代都是从第一行的相应列偏移开始的
                srcTensor_v[offset],
                vmad_mask,
                m_actual,
                vmad_params);

            AscendC::Add<ElementA, true>(
                srcTensor_m,
                srcTensor_m[offset],
                srcTensor_m,
                add_mask,
                m_actual,
                add_params);

            AscendC::PipeBarrier<PIPE_V>();
        }

        if (remain > 0 && repeat_num > 0)
        {
            uint32_t offset = repeat_num * temp_repeat_size;
            if (offset + remain > n_actual)
            {
                remain = n_actual - offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;
            AscendC::Add<ElementA, true>(
                srcTensor_m,
                srcTensor_m[offset],
                srcTensor_m,
                remain_mask,
                m_actual,
                add_params);
            
            AscendC::MulAddDst<ElementAccumulator, ElementA, true>(
                vmad_workspace,
                srcTensor_m[offset], // 这里是起始地址，每次外层迭代都是从第一行的相应列偏移开始的
                srcTensor_v[offset],
                remain_mask,
                m_actual,
                vmad_params);
        }
        
        uint64_t reduce_mask = (repeat_num == 0) ? remain : temp_repeat_size;
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::WholeReduceSum<ElementA, true>(
            srcTensor_m,
            srcTensor_m,
            reduce_mask,
            m_actual,
            1,
            1,
            RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0);
        
        AscendC::WholeReduceSum<ElementAccumulator,true>(
            vmad_workspace, // 目标tensor
            vmad_workspace, // src tensor
            reduce_mask, // 每次repeat 迭代内参与元素的数量，这里一般为256byte对应的列数，即每行上预留的accum空间，除非列数过少，少于256byte对应的列数
            m_actual, // repeat 迭代次数，即只考虑前面m_actual个行作为输出向量的有效值，即矩阵的行数
            1,  // 目标tensor相邻repeat迭代操作间步长，单位为目标tensor元素类型字节数（即目标tensor元素的数量）
            1, // 源操作数单次迭代内相邻datablock的步长，单位为datablock
            8); // 源操作数相邻repeat迭代间的地址步长，即源操作数每次repeat迭代跳过的datablock数目

        add_mask = (m_actual < temp_repeat_size) ? m_actual : temp_repeat_size;
        vmad_mask = (m_actual < temp_repeat_size) ? m_actual : temp_repeat_size;

        add_params.dstRepStride = 8;
        add_params.src0RepStride = 8;
        add_params.src1RepStride = 8;

        vmad_params.dstRepStride = 8;
        vmad_params.src0RepStride = 8;
        vmad_params.src1RepStride = 8; // 这里每次输出8个datablock了，其中元素类型为ElementA，即矩阵精度
        // 将结果写出，即累加到到dstTensor完成运算
        // 此处按照ElementA的精度进行累加

        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Add<ElementA, true>(
            dstTensorSUM,
            srcTensor_m,
            dstTensorSUM,
            add_mask,
            CeilDiv(m_round, temp_repeat_size),
            add_params);

        AscendC::Add<ElementA, true>(
            dstTensorVMAD,
            vmad_workspace,
            dstTensorVMAD,
            vmad_mask,
            CeilDiv(m_round, temp_repeat_size),
            vmad_params);
    }
};

template <>
struct TileFaultSumVmad<Arch::AtlasA2,
                Gemv::helper::FT_REDUCE_TYPE::SUM_VMAD,
                Gemm::GemmType<half, layout::RowMajor>,
                Gemm::GemmType<half, layout::VectorLayout>,
                Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementA = half;
    using ElementX = half;
    using ElementY = float;
    using ElementAccumulator = ElementY;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // Methods

    CATLASS_DEVICE
    TileFaultSumVmad() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensorVMAD,
        AscendC::LocalTensor<ElementY> dstTensorSUM,
        AscendC::LocalTensor<ElementX> srcTensor_v,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        AscendC::LocalTensor<ElementAccumulator> vmad_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);

        uint32_t dstOffset = m_round;

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementAccumulator);

        uint32_t vmad_mask = temp_repeat_size;
        uint32_t repeat_num = n_actual / temp_repeat_size;
        uint32_t remain = n_actual % temp_repeat_size;

        // m_actual *
        uint64_t add_mask = temp_repeat_size;

        /*
        控制操作数地址步长的参数。BinaryRepeatParams类型，
        包含操作数相邻迭代间相同datablock的地址步长，
        操作数同一迭代内不同datablock的地址步长等参数。

        相邻迭代间的地址步长参数说明请参考repeatStride；
        同一迭代内datablock的地址步长参数说明请参考dataBlockStride。
        */

        AscendC::BinaryRepeatParams add_params;
        add_params.dstBlkStride = 1;
        add_params.src0BlkStride = 1;
        add_params.src1BlkStride = 1;
        add_params.dstRepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        add_params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        add_params.src1RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

        // 将输出向量赋值为0
        AscendC::Duplicate<ElementAccumulator>(
            vmad_workspace,
            (ElementAccumulator)0.0,
            // 这是为了后续我们每次求全部行的局部行和，
            // 我们对每行局部和的存储空间设定为256byte对齐，
            // 这是因为每次赋值操作的 repeat 迭代内的运算能力是256byte
            temp_repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
            // 我们的目标是求行和，所以我们的输出空间是行的倍数，且是datablock的倍数
            // 因此在这里，我们在赋值0时通过每次内部迭代对应一个输出行的局部和存储与accum空间
            CeilDiv(m_round * temp_repeat_size, temp_repeat_size), // 求行和
            1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
            8 // 相邻赋值迭代间，矢量目的操作数不同 datablock chunk 间地址步长。
        ); // 总的来说，我们开辟了256 byte x m_round的空间，全部赋值为0，每256byte对应一个行的局部和存储与accum

        AscendC::PipeBarrier<PIPE_V>();

        uint32_t vmad_offset_raw = 0 * temp_repeat_size;
        AscendC::BinaryRepeatParams vmad_params;
        vmad_params.dstBlkStride = 1;
        vmad_params.src0BlkStride = 1;
        vmad_params.src1BlkStride = 1;

        vmad_params.dstRepStride = RoundUp(temp_repeat_size, temp_repeat_size) / (BYTE_PER_C0 / sizeof(ElementAccumulator));
        vmad_params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        vmad_params.src1RepStride = 0;

        if(repeat_num > 0){
            AscendC::MulAddDst<ElementAccumulator, ElementA, true>(
                vmad_workspace,
                srcTensor_m[vmad_offset_raw], // 这里是起始地址，每次外层迭代都是从第一行的相应列偏移开始的
                srcTensor_v[vmad_offset_raw],
                vmad_mask,
                m_actual,
                vmad_params);
        }else{
            // 如果没有完整的repeat_size，那么就直接对remain部分进行vmad
            uint64_t remain_mask = remain;
            AscendC::MulAddDst<ElementAccumulator, ElementA, true>(
                vmad_workspace,
                srcTensor_m[vmad_offset_raw], // 这里是起始地址，每次外层迭代都是从第一行的相应列偏移开始的
                srcTensor_v[vmad_offset_raw],
                remain_mask,
                m_actual,
                vmad_params);
        }

        AscendC::PipeBarrier<PIPE_V>();

        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Add(const LocalTensor<T>& dstLocal, 
        const LocalTensor<T>& src0Local, 
        const LocalTensor<T>& src1Local, 
        uint64_t mask, 
        const uint8_t repeatTimes, 
        const BinaryRepeatParams& repeatParams)
        */

        for (uint32_t i = 1; i < repeat_num; i++) {

            uint32_t offset = i * temp_repeat_size;

            AscendC::MulAddDst<ElementAccumulator, ElementA, true>(
                vmad_workspace,
                srcTensor_m[offset], // 这里是起始地址，每次外层迭代都是从第一行的相应列偏移开始的
                srcTensor_v[offset],
                vmad_mask,
                m_actual,
                vmad_params);

            AscendC::Add<ElementA, true>(
                srcTensor_m,
                srcTensor_m[offset],
                srcTensor_m,
                add_mask,
                m_actual,
                add_params);

            AscendC::PipeBarrier<PIPE_V>();
        }

        if (remain > 0 && repeat_num > 0)
        {
            uint32_t offset = repeat_num * temp_repeat_size;
            if (offset + remain > n_actual)
            {
                remain = n_actual - offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;
            AscendC::Add<ElementA, true>(
                srcTensor_m,
                srcTensor_m[offset],
                srcTensor_m,
                remain_mask,
                m_actual,
                add_params);
            
            AscendC::MulAddDst<ElementAccumulator, ElementA, true>(
                vmad_workspace,
                srcTensor_m[offset], // 这里是起始地址，每次外层迭代都是从第一行的相应列偏移开始的
                srcTensor_v[offset],
                remain_mask,
                m_actual,
                vmad_params);
        }
        
        uint64_t reduce_mask = (repeat_num == 0) ? remain : temp_repeat_size;
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::WholeReduceSum<ElementA, true>(
            srcTensor_m,
            srcTensor_m,
            reduce_mask,
            m_actual,
            1,
            1,
            RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0);
        
        AscendC::WholeReduceSum<ElementAccumulator,true>(
            vmad_workspace, // 目标tensor
            vmad_workspace, // src tensor
            reduce_mask, // 每次repeat 迭代内参与元素的数量，这里一般为256byte对应的列数，即每行上预留的accum空间，除非列数过少，少于256byte对应的列数
            m_actual, // repeat 迭代次数，即只考虑前面m_actual个行作为输出向量的有效值，即矩阵的行数
            1,  // 目标tensor相邻repeat迭代操作间步长，单位为目标tensor元素类型字节数（即目标tensor元素的数量）
            1, // 源操作数单次迭代内相邻datablock的步长，单位为datablock
            8); // 源操作数相邻repeat迭代间的地址步长，即源操作数每次repeat迭代跳过的datablock数目

        add_mask = (m_actual < temp_repeat_size) ? m_actual : temp_repeat_size;
        vmad_mask = (m_actual < temp_repeat_size) ? m_actual : temp_repeat_size;

        add_params.dstRepStride = 8;
        add_params.src0RepStride = 8;
        add_params.src1RepStride = 8;

        vmad_params.dstRepStride = 8;
        vmad_params.src0RepStride = 8;
        vmad_params.src1RepStride = 8; // 这里每次输出8个datablock了，其中元素类型为ElementA，即矩阵精度
        // 将结果写出，即累加到到dstTensor完成运算
        // 此处按照ElementA的精度进行累加

        AscendC::PipeBarrier<PIPE_V>();

        /*
        template <typename T1, typename T2, bool isSetMask = true>
        __aicore__ inline void Cast(const LocalTensor<T1>& dstLocal, 
            const LocalTensor<T2>& srcLocal, const RoundMode& round_mode, 
            const uint64_t mask, const uint8_t repeatTimes, 
            const UnaryRepeatParams& repeatParams)
        T1: 目的操作数类型
        T2：源操作数类型
        函数功能
        根据源操作数和目的操作数Tensor的数据类型进行精度转换。
        将源操作数精度转换为目的操作数的精度之后写入目的操作数

        这里是将中间聚合完成的行和向量降精度后存放在输入向量的头部。
        */

        AscendC::Cast<ElementY, ElementA>(
            dstTensorSUM[dstOffset],
            srcTensor_m,
            AscendC::RoundMode::CAST_NONE, m_actual);

        AscendC::PipeBarrier<PIPE_V>();

         /*
            CAST_NONE = 0,  // 在转换有精度损失时表示CAST_RINT模式，不涉及精度损失时表示不舍入
            CAST_RINT,      // rint，四舍六入五成双舍入
            CAST_RINT模式下，若待舍入部分的第一位为0，则不进位；
            若第一位为1且后续位不全为0，则进位；
            若第一位为1且后续位全为0，
            当M（即位数中保留位数的最后一位）的最后一位为0则不进位，当M的最后一位为1则进位。
        */

        AscendC::Add<ElementY, true>(
            dstTensorSUM,
            dstTensorSUM[dstOffset],
            dstTensorSUM,
            (uint64_t)add_mask,
            CeilDiv(m_round, temp_repeat_size),
            add_params);

        AscendC::Add<ElementY, true>(
            dstTensorVMAD,
            vmad_workspace,
            dstTensorVMAD,
            (uint64_t)vmad_mask,
            CeilDiv(m_round, temp_repeat_size),
            vmad_params);        
    }
};


template <>
struct TileFaultSum<Arch::AtlasA2,
                Gemv::helper::FT_REDUCE_TYPE::MAX_MIN_WITH_CAST,
                Gemm::GemmType<float, layout::RowMajor>,
                Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementA = float;
    using ElementX = float;
    using ElementY = float;
    using ElementAccumulator = ElementY;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // Methods

    CATLASS_DEVICE
    TileFaultSum() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensorMin,
        AscendC::LocalTensor<ElementY> dstTensorMax,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        AscendC::LocalTensor<ElementA> minmax_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = repeat_size;
        uint32_t repeat_num = n_actual / repeat_size;
        uint32_t remain = n_actual % repeat_size;

        // m_actual *
        uint64_t min_mask = repeat_size;
        uint64_t max_mask = repeat_size;

        /*
        控制操作数地址步长的参数。BinaryRepeatParams类型，
        包含操作数相邻迭代间相同datablock的地址步长，
        操作数同一迭代内不同datablock的地址步长等参数。

        相邻迭代间的地址步长参数说明请参考repeatStride；
        同一迭代内datablock的地址步长参数说明请参考dataBlockStride。
        */

        AscendC::BinaryRepeatParams max_params;
        max_params.dstBlkStride = 1;
        max_params.src0BlkStride = 1;
        max_params.src1BlkStride = 1;

        max_params.dstRepStride = RoundUp(repeat_size, repeat_size) / (BYTE_PER_C0 / sizeof(ElementAccumulator));
        max_params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        max_params.src1RepStride = RoundUp(repeat_size, repeat_size) / (BYTE_PER_C0 / sizeof(ElementAccumulator));

        AscendC::BinaryRepeatParams min_params;
        min_params.dstBlkStride = 1;
        min_params.src0BlkStride = 1;
        min_params.src1BlkStride = 1;

        // params.dstRepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

        min_params.dstRepStride = RoundUp(repeat_size, repeat_size) / (BYTE_PER_C0 / sizeof(ElementAccumulator));
        min_params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        min_params.src1RepStride = RoundUp(repeat_size, repeat_size) / (BYTE_PER_C0 / sizeof(ElementAccumulator));
        
        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Add(const LocalTensor<T>& dstLocal, 
        const LocalTensor<T>& src0Local, 
        const LocalTensor<T>& src1Local, 
        uint64_t mask, 
        const uint8_t repeatTimes, 
        const BinaryRepeatParams& repeatParams)
        */


        AscendC::Duplicate<ElementA>(
                minmax_workspace,
                (ElementA)0.0,
                repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
                CeilDiv(m_actual * repeat_size, repeat_size), // 求行最小值
                1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
                8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        );

        uint32_t aim_m_actual = CeilDiv(m_actual * repeat_size, repeat_size);
        uint32_t aim_work_offset = aim_m_actual * repeat_size;

        AscendC::Duplicate<ElementA>(
                minmax_workspace[aim_work_offset],
                (ElementA)0.0,
                repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
                CeilDiv(m_actual * repeat_size, repeat_size), // 求行最大值
                1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
                8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        );

        AscendC::PipeBarrier<PIPE_V>();
        
        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Min(
            const LocalTensor<T>& dstLocal, 
            const LocalTensor<T>& src0Local, 
            const LocalTensor<T>& src1Local, uint64_t mask, 
            const uint8_t repeatTimes, 
            const BinaryRepeatParams& repeatParams)
        */
        // AscendC::Min<ElementA, true>(
        //     minmax_workspace,
        //     srcTensor_m,
        //     minmax_workspace,
        //     min_mask,
        //     m_actual,
        //     min_params);
        
        // AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t i = 0; i < repeat_num; i++) {
            uint32_t offset = i * repeat_size;

            AscendC::Min<ElementA, true>(
                minmax_workspace,
                srcTensor_m[offset],
                minmax_workspace,
                min_mask,
                m_actual,
                min_params);
            
            AscendC::Max<ElementA, true>(
                minmax_workspace[aim_work_offset],
                srcTensor_m[offset],
                minmax_workspace[aim_work_offset],
                max_mask,
                m_actual,
                max_params);
                
            AscendC::PipeBarrier<PIPE_V>();
        }

        if (remain > 0)
        {
            uint32_t offset = repeat_num * repeat_size;
            if (offset + remain > n_actual)
            {
                remain = n_actual - offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;
            AscendC::Min<ElementA, true>(
                minmax_workspace,
                srcTensor_m[offset],
                minmax_workspace,
                remain_mask,
                m_actual,
                min_params);
            
            AscendC::Max<ElementA, true>(
                minmax_workspace[aim_work_offset],
                srcTensor_m[offset],
                minmax_workspace[aim_work_offset],
                remain_mask,
                m_actual,
                max_params);
        }
        
        uint64_t reduce_mask = (repeat_num == 0) ? remain : repeat_size;
        AscendC::PipeBarrier<PIPE_V>();
        
        AscendC::WholeReduceMax<ElementA, true>(
            minmax_workspace[aim_work_offset],
            minmax_workspace[aim_work_offset],
            reduce_mask,
            m_actual,
            1,
            1,
            RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE);

        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void WholeReduceMin(const LocalTensor<T>& dstLocal, 
            const LocalTensor<T>& srcLocal, 
            const int32_t mask, const int32_t repeatTimes,
            const int32_t dstRepStride, const int32_t srcBlkStride,
            const int32_t srcRepStride,
            ReduceOrder order = ReduceOrder::ORDER_VALUE_INDEX)

        ORDER_VALUE_INDEX：表示value位于低半部，返回结果存储顺序为[value, index]。
        ORDER_INDEX_VALUE：表示index位于低半部，返回结果存储顺序为[index, value]。
        ORDER_ONLY_VALUE：表示只返回最值，返回结果存储顺序为[value]。
        ORDER_ONLY_INDEX：表示只返回最值索引，返回结果存储顺序为[index]。
        */

        AscendC::WholeReduceMin<ElementA, true>(
            minmax_workspace,
            minmax_workspace,
            reduce_mask,
            m_actual,
            1,
            1,
            RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE);

        uint64_t min_final_mask = (m_actual < repeat_size) ? m_actual : repeat_size;
        uint64_t max_final_mask = (m_actual < repeat_size) ? m_actual : repeat_size;

        min_params.dstRepStride = 8;
        min_params.src0RepStride = 8;
        min_params.src1RepStride = 8;

        max_params.dstRepStride = 8;
        max_params.src0RepStride = 8;
        max_params.src1RepStride = 8;

        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Min<ElementA, true>(
            dstTensorMin,
            minmax_workspace,
            dstTensorMin,
            min_final_mask,
            CeilDiv(m_round, repeat_size),
            min_params);

        AscendC::Max<ElementA, true>(
            dstTensorMax,
            minmax_workspace[aim_work_offset],
            dstTensorMax,
            max_final_mask,
            CeilDiv(m_round, repeat_size),
            max_params);   
    }
};

}
#endif // CATLASS_GEMV_TILE_TILE_FAULT_SUM_HPP_SELF