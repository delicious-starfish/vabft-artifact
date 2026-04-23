
#ifndef CATLASS_GEMM_TILE_COPY_GM_TO_UB_HPP_SELF
#define CATLASS_GEMM_TILE_COPY_GM_TO_UB_HPP_SELF

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "gemm/tile/tile_copy_tla.hpp"
#include "tla/tensor.hpp"

namespace CubeSelf::Gemm::Tile{

/// Partial specialization for AtlasA2, RowMajor in and RowMajor out.
template <class ElementSrc, class ElementDst, class LayoutSrc_, class LayoutDst_>
struct TileCopyTla<Catlass::Arch::AtlasA2, 
    tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc_, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst_, AscendC::TPosition::VECCALC>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc_>::value &&
                     tla::detail::isRowMajor<LayoutDst_>::value>> {
    using LayoutDst = LayoutDst_;
    using LayoutSrc = LayoutSrc_;
    using TensorDst = tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, AscendC::TPosition::VECCALC>;
    using TensorSrc = tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, AscendC::TPosition::GM>;

    static constexpr uint32_t ELE_NUM_PER_BLK = Catlass::BYTE_PER_BLK / sizeof(ElementSrc);

    // Mehtods

    CATLASS_DEVICE
    TileCopyTla() {};

    CATLASS_DEVICE
    void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        AscendC::DataCopyExtParams dataCopyParams(
            tla::get<0>(srcTensor.shape()),
            tla::get<1>(srcTensor.shape()) * sizeof(ElementSrc),
            (tla::get<0>(srcTensor.stride()) - tla::get<1>(srcTensor.shape())) * sizeof(ElementSrc),
            (tla::get<0>(dstTensor.stride()) - tla::get<1>(dstTensor.shape())) / ELE_NUM_PER_BLK,
            0
        );
        AscendC::DataCopyPadExtParams<ElementSrc> padParams(false, 0, 0, 0);
        AscendC::DataCopyPad(dstTensor.data(), srcTensor.data(), dataCopyParams, padParams);
    };
};

} // CubeSelf::Gemm::Tile

#endif