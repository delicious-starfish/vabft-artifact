
#ifndef CATLASS_GEMM_TILE_COPY_L1_TO_L0B_HPP_SELF
#define CATLASS_GEMM_TILE_COPY_L1_TO_L0B_HPP_SELF

// catlass/

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "gemm/tile/tile_copy_tla.hpp"
#include "tla/tensor.hpp"


namespace CubeSelf::Gemm::Tile{

template <
    class ArchTag,
    class L1Type,
    class L0Type = void
>
struct CopyL1ToL0B {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to l0, can not find the specialization.");
};

////////////////////////////////////////
/// new add gemm
template<class ArchTag, class Element>
struct CopyL1ToL0B<ArchTag,
    Catlass::Gemm::GemmType<Element, Catlass::layout::zZ, AscendC::TPosition::B1>,
    Catlass::Gemm::GemmType<Element, Catlass::layout::nZ, AscendC::TPosition::B2>>
{
    using LayoutDst = Catlass::layout::nZ;
    using LayoutSrc = Catlass::layout::zZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);

    CATLASS_DEVICE
    CopyL1ToL0B(){}

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::LocalTensor<Element> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<ELE_NUM_PER_C0>(layoutSrc.orgShape(1)));
        loadDataParams.srcStride = 1;

        loadDataParams.sid = 0;
        loadDataParams.dstGap = 0;

        loadDataParams.ifTranspose = true;
        loadDataParams.addrMode = 0;

        for(uint32_t i = 0; i < CeilDiv<Catlass::C0_NUM_PER_FRACTAL>(layoutDst.orgShape(0)); i++){  // K N
            AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], 
                srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};

template<class ArchTag>
struct CopyL1ToL0B<ArchTag, 
    Catlass::Gemm::GemmType<float, Catlass::layout::zZ, AscendC::TPosition::B1>,
    Catlass::Gemm::GemmType<float, Catlass::layout::nZ, AscendC::TPosition::B2>>
{

    using Element = float;
    using LayoutDst = Catlass::layout::nZ;
    using LayoutSrc = Catlass::layout::zZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);

    CATLASS_DEVICE
    CopyL1ToL0B(){}

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::LocalTensor<Element> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        AscendC::LoadData2dTransposeParams loadDataParams;
        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<Catlass::C0_NUM_PER_FRACTAL>(layoutSrc.orgShape(1)));

        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;

        loadDataParams.dstFracGap = static_cast<uint16_t>(CeilDiv<Catlass::C0_NUM_PER_FRACTAL>(layoutDst.orgShape(1))) - 1;

        for(uint32_t i = 0; i < CeilDiv<Catlass::C0_NUM_PER_FRACTAL>(layoutDst.orgShape(0)); i++){ // K N
            AscendC::LoadDataWithTranspose(dstTensor[i * layoutDst.stride(1) * 2], 
                srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};

template<class ArchTag>
struct CopyL1ToL0B<ArchTag,
    Catlass::Gemm::GemmType<int8_t, Catlass::layout::zN, AscendC::TPosition::B1>,
    Catlass::Gemm::GemmType<int8_t, Catlass::layout::nZ, AscendC::TPosition::B2>>
{
    using Element = int8_t;
    using LayoutDst = Catlass::layout::nZ;
    using LayoutSrc = Catlass::layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 =  Catlass::BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = Catlass::BYTE_PER_FRACTAL / sizeof(Element);

    CATLASS_DEVICE
    CopyL1ToL0B(){}

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::LocalTensor<Element> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        AscendC::LoadData2dTransposeParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)));
        loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL / 2;
        loadDataParams.dstGap = 1;
        loadDataParams.dstFracGap = 0;

        for (uint32_t i = 0; i < CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(0)); i++) {
            AscendC::LoadDataWithTranspose(dstTensor[i * layoutDst.stride(1)],
                                           srcTensor[i * layoutSrc.stride(1) * 2],
                                           loadDataParams);
        }
    }
};

template<class ArchTag, class Element>
struct CopyL1ToL0B<ArchTag,
    Catlass::Gemm::GemmType<Element, Catlass::layout::nZ, AscendC::TPosition::B1>,
    Catlass::Gemm::GemmType<Element, Catlass::layout::nZ, AscendC::TPosition::B2>>
{
    using LayoutDst = Catlass::layout::nZ;
    using LayoutSrc = Catlass::layout::nZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = Catlass::BYTE_PER_FRACTAL / sizeof(Element);

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0B() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(3));
        loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL;

        loadDataParams.sid = 0;
        loadDataParams.dstGap = layoutDst.stride(3) / ELE_NUM_PER_FRACTAL - 1;
        loadDataParams.ifTranspose = false;

        loadDataParams.addrMode = 0;

        for (uint32_t i = 0; i < layoutDst.shape(1); i++) {
            AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], 
                srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};

template<class ArchTag, class Element>
struct CopyL1ToL0B<ArchTag,
    Catlass::Gemm::GemmType<Element, Catlass::layout::nZ, AscendC::TPosition::A1>,
    Catlass::Gemm::GemmType<Element, Catlass::layout::nZ, AscendC::TPosition::B2>>
{
    using LayoutDst = Catlass::layout::nZ;
    using LayoutSrc = Catlass::layout::nZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = Catlass::BYTE_PER_FRACTAL / sizeof(Element);

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0B() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(3));
        loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL;

        loadDataParams.sid = 0;
        loadDataParams.dstGap = layoutDst.stride(3) / ELE_NUM_PER_FRACTAL - 1;
        loadDataParams.ifTranspose = false;

        loadDataParams.addrMode = 0;

        for (uint32_t i = 0; i < layoutDst.shape(1); i++) {
            AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], 
                srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};

/////////////////////////////////////////////

////////////////////////////////////////////
/// new add gemv
template<class ArchTag, class Element>
struct CopyL1ToL0B<ArchTag, 
    Catlass::Gemm::GemmType<Element, Catlass::layout::zN, AscendC::TPosition::B1>,
    Catlass::Gemm::GemmType<Element, Catlass::layout::zN, AscendC::TPosition::B2>>
{
    using LayoutDst = Catlass::layout::zN;
    using LayoutSrc = Catlass::layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = Catlass::BYTE_PER_FRACTAL / sizeof(Element);

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0B() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(1));
        loadDataParams.srcStride = layoutSrc.stride(1) / ELE_NUM_PER_FRACTAL;

        loadDataParams.sid = 0;
        loadDataParams.dstGap = layoutDst.stride(1) / ELE_NUM_PER_FRACTAL - 1;
        loadDataParams.ifTranspose = false;

        loadDataParams.addrMode = 0;

        for (uint32_t i = 0; i < layoutDst.shape(3); i++)
        {
            AscendC::LoadData(dstTensor[i * layoutDst.stride(3)], 
                srcTensor[i * layoutSrc.stride(3)], loadDataParams);
        }
    }
};

template<class ArchTag, class Element>
struct CopyL1ToL0B<ArchTag, 
    Catlass::Gemm::GemmType<Element, Catlass::layout::nN, AscendC::TPosition::B1>,
    Catlass::Gemm::GemmType<Element, Catlass::layout::zN, AscendC::TPosition::B2>>
{
    using LayoutDst = Catlass::layout::zN;
    using LayoutSrc = Catlass::layout::nN;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = Catlass::BYTE_PER_FRACTAL / sizeof(Element);

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0B() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = layoutDst.shape(1) * layoutDst.shape(3);
        loadDataParams.srcStride = layoutSrc.stride(1) / ELE_NUM_PER_FRACTAL;

        loadDataParams.sid = 0;
        loadDataParams.dstGap = layoutDst.stride(1) / ELE_NUM_PER_FRACTAL - 1;
        loadDataParams.ifTranspose = true;

        loadDataParams.addrMode = 0;

        AscendC::LoadData(dstTensor, srcTensor, loadDataParams);
    };
};

template<class ArchTag>
struct CopyL1ToL0B<ArchTag,
    Catlass::Gemm::GemmType<float, Catlass::layout::nN, AscendC::TPosition::B1>,
    Catlass::Gemm::GemmType<float, Catlass::layout::zN, AscendC::TPosition::B2>>
{
    using LayoutDst = Catlass::layout::zN;
    using LayoutSrc = Catlass::layout::nN;
    using Element = float;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = Catlass::BYTE_PER_FRACTAL / sizeof(Element);

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0B() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2dTransposeParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<Catlass::C0_NUM_PER_FRACTAL>(layoutDst.orgShape(0)));
        loadDataParams.srcStride = 1;

        loadDataParams.dstGap = 0;
        loadDataParams.dstFracGap = CeilDiv<Catlass::C0_NUM_PER_FRACTAL>(layoutDst.orgShape(0)) - 1;

        for (uint32_t i = 0; i < CeilDiv<2 * ELE_NUM_PER_C0>(layoutDst.orgShape(1)); i++)
        {
            AscendC::LoadDataWithTranspose(
                dstTensor[i * layoutDst.stride(3) * 2],
                srcTensor[i * layoutSrc.stride(3)],
                loadDataParams);
        }
    };
};

template<class ArchTag>
struct CopyL1ToL0B<ArchTag, 
    Catlass::Gemm::GemmType<int8_t, Catlass::layout::nZ, AscendC::TPosition::B1>,
    Catlass::Gemm::GemmType<int8_t, Catlass::layout::zN, AscendC::TPosition::B2>>
{
    using LayoutDst = Catlass::layout::zN;
    using LayoutSrc = Catlass::layout::nZ;
    using Element = int8_t;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = Catlass::BYTE_PER_FRACTAL / sizeof(Element);

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0B() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2dTransposeParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(0)));
        loadDataParams.srcStride = layoutSrc.stride(1) / ELE_NUM_PER_FRACTAL / 2;
        loadDataParams.dstGap = 1;
        loadDataParams.dstFracGap = 0;

        for (uint32_t i = 0; i < CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)); i++)
        {
            AscendC::LoadDataWithTranspose(
                dstTensor[i * layoutDst.stride(3)],
                srcTensor[i * layoutSrc.stride(3) * 2],
                loadDataParams);
        }
    }
};

////////////////////////////////////////////

/// Partial specialization for int8_t, zN in and nZ out.
template <class ArchTag>
struct CopyL1ToL0B<ArchTag, 
    Catlass::Gemm::GemmType<int8_t, Catlass::layout::zN, AscendC::TPosition::A1>>
{
    using Element = int8_t;
    using LayoutDst = Catlass::layout::nZ;
    using LayoutSrc = Catlass::layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = Catlass::BYTE_PER_FRACTAL / sizeof(Element);

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0B() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2dTransposeParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)));
        loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL / 2;
        loadDataParams.dstGap = 1;
        loadDataParams.dstFracGap = 0;

        for (uint32_t i = 0; i < CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(0)); i++) {
            AscendC::LoadDataWithTranspose(dstTensor[i * layoutDst.stride(1)],
                                           srcTensor[i * layoutSrc.stride(1) * 2],
                                           loadDataParams);
        }
    }
};

/// Partial specialization for float, zN in and nZ out.
template<class ArchTag>
struct CopyL1ToL0B<ArchTag, 
    Catlass::Gemm::GemmType<float, Catlass::layout::zN, AscendC::TPosition::A1>>
{
    using Element = float;
    using LayoutDst = Catlass::layout::nZ;
    using LayoutSrc = Catlass::layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = Catlass::BYTE_PER_FRACTAL / sizeof(Element);

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0B() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        constexpr uint8_t PAD_LIST[4] = {0, 0, 0, 0};
        uint16_t l1K = layoutSrc.shape(0) * layoutSrc.shape(1);
        uint16_t l1N = layoutSrc.shape(2) * layoutSrc.shape(3);
        uint16_t l0K = layoutDst.shape(0) * layoutDst.shape(1);
        uint16_t l0N = layoutDst.shape(2) * layoutDst.shape(3);
        // K, N need to be 16 aligned for f32
        uint16_t l1KAlign = RoundUp<Catlass::C0_NUM_PER_FRACTAL>(l1K);
        uint16_t l1NAlign = RoundUp<Catlass::C0_NUM_PER_FRACTAL>(l1N);
        uint16_t l0KAlign = RoundUp<Catlass::C0_NUM_PER_FRACTAL>(l0K);
        uint16_t l0NAlign = RoundUp<Catlass::C0_NUM_PER_FRACTAL>(l0N);
        AscendC::SetFmatrix(1, l1KAlign, PAD_LIST, AscendC::FmatrixMode::FMATRIX_RIGHT);
        static constexpr AscendC::IsResetLoad3dConfig config = {false, false};
        AscendC::LoadData3DParamsV2<Element> loadDataParams;
        loadDataParams.kExtension = l0NAlign;
        loadDataParams.mExtension = l0KAlign;
        loadDataParams.channelSize = l1NAlign;
        loadDataParams.fMatrixCtrl = true;

        AscendC::LoadData<Element, config>(dstTensor, srcTensor, loadDataParams);
    }
};

/// Partial specialization for zN in and nZ out.
template<class ArchTag, class Element>
struct CopyL1ToL0B<ArchTag, 
    Catlass::Gemm::GemmType<Element, Catlass::layout::zN, AscendC::TPosition::A1>>
{
    using LayoutDst = Catlass::layout::nZ;
    using LayoutSrc = Catlass::layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = Catlass::BYTE_PER_FRACTAL / sizeof(Element);

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0B() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)));
        loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = layoutDst.stride(3) / ELE_NUM_PER_FRACTAL - 1;
        loadDataParams.ifTranspose = true;
        loadDataParams.addrMode = 0;

        for (uint32_t i = 0; i < CeilDiv<Catlass::C0_NUM_PER_FRACTAL>(layoutDst.orgShape(0)); i++) {
            AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], 
                srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};

/// Partial specialization for nZ in and nZ out. (Transpose B)
template<class ArchTag, class Element>
struct CopyL1ToL0B<ArchTag, 
    Catlass::Gemm::GemmType<Element, Catlass::layout::nZ, AscendC::TPosition::A1>>
{
    using LayoutDst = Catlass::layout::nZ;
    using LayoutSrc = Catlass::layout::nZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = Catlass::BYTE_PER_FRACTAL / sizeof(Element);

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0B() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;
        if (layoutSrc.shape(3) == layoutDst.shape(3)) {
            loadDataParams.startIndex = 0;
            loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(1) * layoutDst.shape(3));
            loadDataParams.srcStride = 1;
            loadDataParams.sid = 0;
            loadDataParams.dstGap = 0;
            loadDataParams.ifTranspose = false;
            loadDataParams.addrMode = 0;

            AscendC::LoadData(dstTensor, srcTensor, loadDataParams);
        } else {
            loadDataParams.startIndex = 0;
            loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(3));
            loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL;
            loadDataParams.sid = 0;
            loadDataParams.dstGap = layoutDst.stride(3) / ELE_NUM_PER_FRACTAL - 1;
            loadDataParams.ifTranspose = false;
            loadDataParams.addrMode = 0;

            for (uint32_t i = 0; i < layoutDst.shape(1); i++) {
                AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
            }
        }

    }
};

///////////////////////////////////////////TileCopyTla//////////////////////////////////////////////////////
/// Partial specialization for CopyL1ToL0B, AtlasA2, zN in and nZ out.
template <class ElementSrc, class ElementDst, class LayoutSrc_, class LayoutDst_>
struct TileCopyTla<Catlass::Arch::AtlasA2,
    tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc_, AscendC::TPosition::A1>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst_, AscendC::TPosition::B2>,
    std::enable_if_t<tla::detail::isnZ<ElementDst, LayoutDst_>::value &&
                     tla::detail::iszN<ElementSrc, LayoutSrc_>::value>> {
    using LayoutDst = LayoutDst_;
    using LayoutSrc = LayoutSrc_;
    using TensorDst = tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, AscendC::TPosition::B2>;
    using TensorSrc = tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, AscendC::TPosition::A1>;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(ElementSrc);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = Catlass::BYTE_PER_FRACTAL / sizeof(ElementSrc);

    // Mehtods

    CATLASS_DEVICE
    TileCopyTla() {};

    CATLASS_DEVICE
    void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        const uint32_t srcOuterStrideRow = tla::get<0, 1>(srcTensor.stride());
        const uint32_t srcOuterStrideCol = tla::get<1, 1>(srcTensor.stride());
        const uint32_t dstOuterShapeRow = tla::get<0, 1>(dstTensor.shape());
        const uint32_t dstOuterShapeCol = tla::get<1, 1>(dstTensor.shape());
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());

        AscendC::LoadData2DParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = dstOuterShapeCol;
        loadDataParams.srcStride = srcOuterStrideCol / ELE_NUM_PER_FRACTAL;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = true;
        loadDataParams.addrMode = 0;

        for (uint32_t i = 0; i < dstOuterShapeRow; i++) {
            AscendC::LoadData(dstTensor.data()[i * dstOuterStrideRow],
                              srcTensor.data()[i * srcOuterStrideRow],
                              loadDataParams);
        }
    }
};

/// Partial specialization for CopyL1ToL0B, AtlasA2, nZ in and nZ out. (Transpose B)
template <class ElementSrc, class ElementDst, class LayoutSrc_, class LayoutDst_>
struct TileCopyTla<Catlass::Arch::AtlasA2, 
    tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc_, AscendC::TPosition::A1>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst_, AscendC::TPosition::B2>,
    std::enable_if_t<tla::detail::isnZ<ElementDst, LayoutDst_>::value &&
                     tla::detail::isnZ<ElementSrc, LayoutSrc_>::value>> {
    using LayoutDst = LayoutDst_;
    using LayoutSrc = LayoutSrc_;
    using TensorDst = tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, AscendC::TPosition::B2>;
    using TensorSrc = tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, AscendC::TPosition::A1>;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(ElementSrc);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = Catlass::BYTE_PER_FRACTAL / sizeof(ElementSrc);

    // Mehtods

    CATLASS_DEVICE
    TileCopyTla() {};

    CATLASS_DEVICE
    void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        const uint32_t srcOuterStrideRow = tla::get<0, 1>(srcTensor.stride());
        const uint32_t srcOuterStrideCol = tla::get<1, 1>(srcTensor.stride());
        const uint32_t dstOuterShapeRow = tla::get<0, 1>(dstTensor.shape());
        const uint32_t dstOuterShapeCol = tla::get<1, 1>(dstTensor.shape());
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());

        AscendC::LoadData2DParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = dstOuterShapeCol;
        loadDataParams.srcStride = srcOuterStrideCol / ELE_NUM_PER_FRACTAL;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = false;
        loadDataParams.addrMode = 0;

        for (uint32_t i = 0; i < dstOuterShapeRow; i++) {
            AscendC::LoadData(dstTensor.data()[i * dstOuterStrideRow],
                              srcTensor.data()[i * srcOuterStrideRow],
                              loadDataParams);
        }
    }
};

/// Partial specialization for CopyL1ToL0B, AtlasA2, int8_t, zN in and nZ out.
template <class LayoutSrc_, class LayoutDst_>
struct TileCopyTla<Catlass::Arch::AtlasA2, 
    tla::Tensor<AscendC::LocalTensor<int8_t>, LayoutSrc_, AscendC::TPosition::A1>,
    tla::Tensor<AscendC::LocalTensor<int8_t>, LayoutDst_, AscendC::TPosition::B2>,
    std::enable_if_t<tla::detail::isnZ<int8_t, LayoutDst_>::value &&
                     tla::detail::iszN<int8_t, LayoutSrc_>::value>> {
    using Element = int8_t;
    using LayoutDst = LayoutDst_;
    using LayoutSrc = LayoutSrc_;
    using TensorDst = tla::Tensor<AscendC::LocalTensor<Element>, LayoutDst, AscendC::TPosition::B2>;
    using TensorSrc = tla::Tensor<AscendC::LocalTensor<Element>, LayoutSrc, AscendC::TPosition::A1>;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = Catlass::BYTE_PER_FRACTAL / sizeof(Element);

    // Mehtods

    CATLASS_DEVICE
    TileCopyTla() {};

    CATLASS_DEVICE
    void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        const uint32_t srcOuterShapeCol = tla::get<1, 1>(srcTensor.shape());
        const uint32_t srcOuterStrideRow = tla::get<0, 1>(srcTensor.stride());
        const uint32_t srcOuterStrideCol = tla::get<1, 1>(srcTensor.stride());
        const uint32_t dstOuterShapeRow = tla::get<0, 1>(dstTensor.shape());
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());

        AscendC::LoadData2dTransposeParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = srcOuterShapeCol;
        loadDataParams.srcStride = srcOuterStrideCol / ELE_NUM_PER_FRACTAL / 2;
        loadDataParams.dstGap = 1;
        loadDataParams.dstFracGap = 0;

        for (uint32_t i = 0; i < dstOuterShapeRow; i++) {
            AscendC::LoadDataWithTranspose(dstTensor.data()[i * dstOuterStrideRow],
                                           srcTensor.data()[i * srcOuterStrideRow * 2],
                                           loadDataParams);
        }
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace CubeSelf::Gemm::Tile

#endif