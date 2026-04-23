#ifndef CATLASS_GEMV_TILE_TILE_VECTOR_ELEM_ADD_HPP_SELF
#define CATLASS_GEMV_TILE_TILE_VECTOR_ELEM_ADD_HPP_SELF

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
    Gemv::helper::VEC_ADD_TYPE VecAddType,
    class AType,
    class YType,
    class BiasType = void
>
struct TileVectorAdd
{
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported TileVectorAdd, can not find the specialization.");
};

template <
    class ElementA,
    class ElementY
>
struct TileVectorAdd<Arch::AtlasA2,
                Gemv::helper::VEC_ADD_TYPE::COUNT,
                Gemm::GemmType<ElementA, layout::VectorLayout>,
                Gemm::GemmType<ElementY, layout::VectorLayout>,
                void>
{
    static constexpr Gemv::helper::VEC_ADD_TYPE VecAddType = Gemv::helper::VEC_ADD_TYPE::COUNT;
};

template <
    class ElementA,
    class ElementY
>
struct TileVectorAdd<Arch::AtlasA2,
                Gemv::helper::VEC_ADD_TYPE::MASK,
                Gemm::GemmType<ElementA, layout::VectorLayout>,
                Gemm::GemmType<ElementY, layout::VectorLayout>,
                void>
{
    static constexpr Gemv::helper::VEC_ADD_TYPE VecAddType = Gemv::helper::VEC_ADD_TYPE::MASK;
};



template <>
struct TileVectorAdd<Arch::AtlasA2,
                Gemv::helper::VEC_ADD_TYPE::COUNT,
                Gemm::GemmType<float, layout::VectorLayout>,
                Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementA = float;
    using ElementX = float;
    using ElementY = float;
    using ElementAccumulator = ElementY;
    static constexpr Gemv::helper::VEC_ADD_TYPE VecAddType = Gemv::helper::VEC_ADD_TYPE::COUNT;

    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);
    static constexpr uint32_t MAX_COMPUTE_LENGTH = 32 * 1024 / sizeof(ElementA);

    // Methods

    CATLASS_DEVICE
    TileVectorAdd() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementA> srcTensor_m1,
        AscendC::LocalTensor<ElementA> srcTensor_m2,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t total_ele_num_round = layoutDst.shape(0);
        uint32_t total_ele_num = layoutSrc.shape(0);

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = repeat_size;
        uint32_t repeat_num = total_ele_num / MAX_COMPUTE_LENGTH;
        uint32_t remain = total_ele_num % MAX_COMPUTE_LENGTH;

        for(uint32_t i = 0; i < repeat_num; i++){
            // Do the calculation
            uint32_t offset = i * MAX_COMPUTE_LENGTH;
            AscendC::Add(dstTensor[offset], srcTensor_m1[offset], srcTensor_m2[offset], MAX_COMPUTE_LENGTH);
        }

        if(remain > 0){
            uint32_t remain_offset = repeat_num * MAX_COMPUTE_LENGTH;

            if (remain_offset + remain > total_ele_num)
            {
                remain = total_ele_num - remain_offset;
            }
            AscendC::Add(dstTensor[remain_offset], srcTensor_m1[remain_offset], srcTensor_m2[remain_offset], remain);
        }

        // AscendC::PipeBarrier<PIPE_V>();
    }
};

template <>
struct TileVectorAdd<Arch::AtlasA2,
                Gemv::helper::VEC_ADD_TYPE::MASK,
                Gemm::GemmType<float, layout::VectorLayout>,
                Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementA = float;
    using ElementX = float;
    using ElementY = float;
    using ElementAccumulator = ElementY;
    static constexpr Gemv::helper::VEC_ADD_TYPE VecAddType = Gemv::helper::VEC_ADD_TYPE::MASK;

    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);
    static constexpr uint32_t MAX_REPEAT_TIMES = 128;

    // Methods

    CATLASS_DEVICE
    TileVectorAdd() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementA> srcTensor_m1,
        AscendC::LocalTensor<ElementA> srcTensor_m2,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t total_ele_num_round = layoutDst.shape(0);
        uint32_t total_ele_num = layoutSrc.shape(0);

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = repeat_size;

        uint32_t total_repeat_num = total_ele_num / repeat_size;
        uint32_t repeat_group_num = (total_repeat_num + MAX_REPEAT_TIMES - 1) / MAX_REPEAT_TIMES;

        uint32_t remain = total_ele_num % repeat_size;

        AscendC::BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;
        params.dstRepStride = 8;
        params.src0RepStride = 8;
        params.src1RepStride = 8;

        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Add(const LocalTensor<T>& dstLocal, 
            const LocalTensor<T>& src0Local, 
            const LocalTensor<T>& src1Local, 
            uint64_t mask, 
            const uint8_t repeatTimes, 
            const BinaryRepeatParams& repeatParams)
        */

        for(uint32_t i = 0; i < repeat_group_num; i++){
            // Do the calculation
            uint32_t offset = i * MAX_REPEAT_TIMES * repeat_size;
            uint32_t actualRepeatTime = (i < (repeat_group_num - 1)) ? MAX_REPEAT_TIMES : (total_repeat_num - (i * MAX_REPEAT_TIMES));
            AscendC::Add<ElementA, true>(
                dstTensor[offset],
                srcTensor_m1[offset],
                srcTensor_m2[offset],
                mask,
                actualRepeatTime,
                params);
        }

        if(remain > 0){
            uint32_t remain_offset = total_repeat_num * repeat_size;

            if (remain_offset + remain > total_ele_num)
            {
                remain = total_ele_num - remain_offset;
            }
            AscendC::Add(dstTensor[remain_offset], srcTensor_m1[remain_offset], srcTensor_m2[remain_offset], remain);
        }

        // AscendC::PipeBarrier<PIPE_V>();
    }
};

template <>
struct TileVectorAdd<Arch::AtlasA2,
                Gemv::helper::VEC_ADD_TYPE::COUNT,
                Gemm::GemmType<half, layout::VectorLayout>,
                Gemm::GemmType<half, layout::VectorLayout>,
                void>
{
    using ElementA = half;
    using ElementX = half;
    using ElementY = half;
    using ElementAccumulator = ElementY;

    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);
    static constexpr uint32_t MAX_COMPUTE_LENGTH = 32 * 1024 / sizeof(ElementA);
    static constexpr Gemv::helper::VEC_ADD_TYPE VecAddType = Gemv::helper::VEC_ADD_TYPE::COUNT;

    // Methods

    CATLASS_DEVICE
    TileVectorAdd() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementA> srcTensor_m1,
        AscendC::LocalTensor<ElementA> srcTensor_m2,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t total_ele_num_round = layoutDst.shape(0);
        uint32_t total_ele_num = layoutSrc.shape(0);

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = repeat_size;
        uint32_t repeat_num = total_ele_num / MAX_COMPUTE_LENGTH;
        uint32_t remain = total_ele_num % MAX_COMPUTE_LENGTH;

        for(uint32_t i = 0; i < repeat_num; i++){
            // Do the calculation
            uint32_t offset = i * MAX_COMPUTE_LENGTH;
            AscendC::Add(dstTensor[offset], srcTensor_m1[offset], srcTensor_m2[offset], MAX_COMPUTE_LENGTH);
        }

        if(remain > 0){
            uint32_t remain_offset = repeat_num * MAX_COMPUTE_LENGTH;

            if (remain_offset + remain > total_ele_num)
            {
                remain = total_ele_num - remain_offset;
            }
            AscendC::Add(dstTensor[remain_offset], srcTensor_m1[remain_offset], srcTensor_m2[remain_offset], remain);
        }

        // AscendC::PipeBarrier<PIPE_V>();
    }
};

template <>
struct TileVectorAdd<Arch::AtlasA2,
                Gemv::helper::VEC_ADD_TYPE::MASK,
                Gemm::GemmType<half, layout::VectorLayout>,
                Gemm::GemmType<half, layout::VectorLayout>,
                void>
{
    using ElementA = half;
    using ElementX = half;
    using ElementY = half;
    using ElementAccumulator = ElementY;
    static constexpr Gemv::helper::VEC_ADD_TYPE VecAddType = Gemv::helper::VEC_ADD_TYPE::MASK;

    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);
    static constexpr uint32_t MAX_REPEAT_TIMES = 128;

    // Methods

    CATLASS_DEVICE
    TileVectorAdd() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementA> srcTensor_m1,
        AscendC::LocalTensor<ElementA> srcTensor_m2,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t total_ele_num_round = layoutDst.shape(0);
        uint32_t total_ele_num = layoutSrc.shape(0);

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = repeat_size;

        uint32_t total_repeat_num = total_ele_num / repeat_size;
        uint32_t repeat_group_num = (total_repeat_num + MAX_REPEAT_TIMES - 1) / MAX_REPEAT_TIMES;
        
        uint32_t remain = total_ele_num % repeat_size;

        AscendC::BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;
        params.dstRepStride = 8;
        params.src0RepStride = 8;
        params.src1RepStride = 8;

        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Add(const LocalTensor<T>& dstLocal, 
            const LocalTensor<T>& src0Local, 
            const LocalTensor<T>& src1Local, 
            uint64_t mask, 
            const uint8_t repeatTimes, 
            const BinaryRepeatParams& repeatParams)
        */

        for(uint32_t i = 0; i < repeat_group_num; i++){
            // Do the calculation
            uint32_t offset = i * MAX_REPEAT_TIMES * repeat_size;
            uint32_t actualRepeatTime = (i < (repeat_group_num - 1)) ? MAX_REPEAT_TIMES : (total_repeat_num - (i * MAX_REPEAT_TIMES));
            AscendC::Add<ElementA, true>(
                dstTensor[offset],
                srcTensor_m1[offset],
                srcTensor_m2[offset],
                mask,
                actualRepeatTime,
                params);
        }

        if(remain > 0){
            uint32_t remain_offset = total_repeat_num * repeat_size;

            if (remain_offset + remain > total_ele_num)
            {
                remain = total_ele_num - remain_offset;
            }
            AscendC::Add(dstTensor[remain_offset], srcTensor_m1[remain_offset], srcTensor_m2[remain_offset], remain);
        }
        // AscendC::PipeBarrier<PIPE_V>();
    }
};

template <>
struct TileVectorAdd<Arch::AtlasA2,
                Gemv::helper::VEC_ADD_TYPE::COUNT,
                Gemm::GemmType<half, layout::VectorLayout>,
                Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementA = half;
    using ElementX = half;
    using ElementY = float;
    using ElementAccumulator = ElementY;

    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);
    static constexpr uint32_t MAX_COMPUTE_LENGTH = 32 * 1024 / sizeof(ElementA);
    static constexpr Gemv::helper::VEC_ADD_TYPE VecAddType = Gemv::helper::VEC_ADD_TYPE::COUNT;

    // Methods

    CATLASS_DEVICE
    TileVectorAdd() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementA> srcTensor_m1,
        AscendC::LocalTensor<ElementA> srcTensor_m2,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t total_ele_num_round = layoutDst.shape(0);
        uint32_t total_ele_num = layoutSrc.shape(0);

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = repeat_size;
        uint32_t repeat_num = total_ele_num / MAX_COMPUTE_LENGTH;
        uint32_t remain = total_ele_num % MAX_COMPUTE_LENGTH;

        for(uint32_t i = 0; i < repeat_num; i++){
            // Do the calculation
            uint32_t offset = i * MAX_COMPUTE_LENGTH;
            AscendC::Add(srcTensor_m1[offset], srcTensor_m1[offset], srcTensor_m2[offset], MAX_COMPUTE_LENGTH);

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Cast<ElementY, ElementA>(
                dstTensor[offset],
                srcTensor_m1[offset],
                AscendC::RoundMode::CAST_NONE, 
                MAX_COMPUTE_LENGTH);
        }

        if(remain > 0){
            uint32_t remain_offset = repeat_num * MAX_COMPUTE_LENGTH;

            if (remain_offset + remain > total_ele_num)
            {
                remain = total_ele_num - remain_offset;
            }
            
            AscendC::Add(srcTensor_m1[remain_offset], srcTensor_m1[remain_offset], srcTensor_m2[remain_offset], remain);

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Cast<ElementY, ElementA>(
                dstTensor[remain_offset],
                srcTensor_m1[remain_offset],
                AscendC::RoundMode::CAST_NONE, 
                remain);
        }   
    }
};

}
#endif // CATLASS_GEMV_TILE_TILE_FAULT_SUM_HPP_SELF