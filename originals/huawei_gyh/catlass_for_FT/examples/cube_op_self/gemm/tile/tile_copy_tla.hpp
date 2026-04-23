
#ifndef CATLASS_GEMM_TILE_TILE_COPY_TLA_HPP_SELF
#define CATLASS_GEMM_TILE_TILE_COPY_TLA_HPP_SELF

#include "catlass/catlass.hpp"

namespace CubeSelf::Gemm::Tile{

template<
    class ArchTag,
    class TensorSrc,
    class TensorDst,
    class Enable = void
>
struct TileCopyTla {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported TileCopyTla, can not find the specialization.");
};

// Extended template for TileCopyTla that supports manually specifying LayoutTagSrc and LayoutTagDst.
// Users can specialize the copy class by LayoutTagSrc and LayoutTagDst.
template<
    class ArchTag,
    class TensorSrc,
    class TensorDst,
    class LayoutTagSrc,
    class LayoutTagDst
>
struct TileCopyTlaExt {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported TileCopyTlaExt, can not find the specialization.");
};

} // CubeSelf::Gemm::Tile


#endif