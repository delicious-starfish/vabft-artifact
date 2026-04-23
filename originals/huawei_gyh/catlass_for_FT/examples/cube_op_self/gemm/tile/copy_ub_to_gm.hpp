#ifndef CATLASS_GEMM_TILE_COPY_UB_TO_GM_HPP
#define CATLASS_GEMM_TILE_COPY_UB_TO_GM_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "gemm/tile/tile_copy_tla.hpp"
#include "tla/tensor.hpp"

namespace CubeSelf::Gemm::Tile {
/// Partial specialization for AtlasA2, RowMajor in and RowMajor out.
template <class ElementSrc, class ElementDst, class LayoutSrc_, class LayoutDst_>
struct TileCopyTla<Catlass::Arch::AtlasA2, 
    tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc_, AscendC::TPosition::VECCALC>,
    tla::Tensor<AscendC::GlobalTensor<ElementDst>, LayoutDst_, AscendC::TPosition::GM>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc_>::value &&
                     tla::detail::isRowMajor<LayoutDst_>::value>> {
    using LayoutDst = LayoutDst_;
    using LayoutSrc = LayoutSrc_;
    using TensorDst = tla::Tensor<AscendC::GlobalTensor<ElementDst>, LayoutDst, AscendC::TPosition::GM>;
    using TensorSrc = tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, AscendC::TPosition::VECCALC>;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(ElementSrc);

    // Mehtods

    CATLASS_DEVICE
    TileCopyTla() {};

    CATLASS_DEVICE
    void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        AscendC::DataCopyExtParams dataCopyParams(
            tla::get<0>(dstTensor.shape()),
            tla::get<1>(dstTensor.shape()) * sizeof(ElementSrc),
            (tla::get<0>(srcTensor.stride()) - tla::get<1>(srcTensor.shape())) / ELE_NUM_PER_C0,
            (tla::get<0>(dstTensor.stride()) - tla::get<1>(dstTensor.shape())) * sizeof(ElementSrc),
            0
        );
        AscendC::DataCopyPad(dstTensor.data(), srcTensor.data(), dataCopyParams);
    };
};

/// Partial specialization for AtlasA2, RowMajor in and PaddingRowMajor out.
template <class ElementSrc, class ElementDst, class LayoutSrc_, class LayoutDst_>
struct TileCopyTlaExt<Catlass::Arch::AtlasA2, 
    tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc_, AscendC::TPosition::VECCALC>,
    tla::Tensor<AscendC::GlobalTensor<ElementDst>, LayoutDst_, AscendC::TPosition::GM>,
    Catlass::layout::RowMajor, Catlass::layout::PaddingRowMajor> {
    using LayoutDst = LayoutDst_;
    using LayoutSrc = LayoutSrc_;
    using TensorDst = tla::Tensor<AscendC::GlobalTensor<ElementDst>, LayoutDst, AscendC::TPosition::GM>;
    using TensorSrc = tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, AscendC::TPosition::VECCALC>;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(ElementSrc);

    // Mehtods

    CATLASS_DEVICE
    TileCopyTlaExt() {};

    CATLASS_DEVICE
    void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        AscendC::DataCopyExtParams dataCopyParams(
            tla::get<1, 1>(dstTensor.shape()),
            tla::get<1, 0>(dstTensor.shape()) * sizeof(ElementSrc),
            (tla::get<0>(srcTensor.stride()) - tla::get<1>(srcTensor.shape())) / ELE_NUM_PER_C0,
            (tla::get<1, 1>(dstTensor.stride()) - tla::get<1, 0>(dstTensor.shape())) * sizeof(ElementSrc),
            0
        );
        AscendC::DataCopyPad(dstTensor.data(), srcTensor.data(), dataCopyParams);
    };
};

} // namespace CubeSelf::Gemm::Tile
#endif