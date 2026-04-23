#ifndef CATLASS_GEMV_TILE_VMULS_HPP_SELF
#define CATLASS_GEMV_TILE_VMULS_HPP_SELF

# include "catlass/catlass.hpp"
# include "catlass/layout/layout.hpp"

namespace Catlass::Gemv::Tile{

template <
    class ArchTag,
    class VType_
>
struct TileVmuls
{
    using Element = typename VType_::Element;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    CATLASS_DEVICE
    TileVmuls() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::LocalTensor<Element> srcTensor,
        Element scalar,
        uint32_t len)
    {
    AscendC::SetMaskCount();
    AscendC::SetVectorMask<Element, AscendC::MaskMode::COUNTER>(len);
        AscendC::Muls<Element,false>(
            dstTensor,
            srcTensor,
            scalar,
            AscendC::MASK_PLACEHOLDER,
            1,
            AscendC::UnaryRepeatParams{}
        );
    AscendC::SetMaskNorm();
    AscendC::ResetMask();
    }
};
} // namespace Catlass::Gemv::Tile
#endif