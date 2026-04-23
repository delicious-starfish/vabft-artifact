#ifndef CATLASS_GEMM_TILE_COPY_L1_TO_BT_HPP_SELF
#define CATLASS_GEMM_TILE_COPY_L1_TO_BT_HPP_SELF

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "tla/tensor.hpp"

using namespace tla;

namespace CubeSelf::Gemm::Tile{
template<
    class ArchTag,
    class L1Type,
    class L0Type = void
>
struct CopyL1ToBT {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to biasTable buffer, can not find the specialization.");
};

template<class ArchTag, class ElementSrc, class ElementDst>
struct CopyL1ToBT<ArchTag,
    Catlass::Gemm::GemmType<ElementSrc, Catlass::layout::VectorLayout, AscendC::TPosition::A1>,
    Catlass::Gemm::GemmType<ElementDst, Catlass::layout::VectorLayout, AscendC::TPosition::C2>>
{
    using LayoutDst = Catlass::layout::VectorLayout;
    using LayoutSrc = Catlass::layout::VectorLayout;

    /*
    constexpr uint32_t Catlass::BYTE_PER_C2 = 64; -> 向C2传输数据时 datablock 的规模为64byte，即单位为64 byte
    */
    static constexpr uint32_t ELE_NUM_PER_C2 =  Catlass::BYTE_PER_C2 / sizeof(ElementSrc);

    CATLASS_DEVICE
    CopyL1ToBT(){}

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementDst> dstTensor,
        AscendC::LocalTensor<ElementSrc> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        AscendC::DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = layoutDst.shape(0) / ELE_NUM_PER_C2;
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        AscendC::DataCopy(dstTensor, srcTensor, intriParams);
    }
};
}

#endif