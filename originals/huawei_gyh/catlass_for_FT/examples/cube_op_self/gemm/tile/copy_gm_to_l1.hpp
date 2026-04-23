#ifndef CATLASS_GEMM_TILE_COPY_GM_TO_L1_HPP_SELF
#define CATLASS_GEMM_TILE_COPY_GM_TO_L1_HPP_SELF

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "gemm/tile/tile_copy_tla.hpp"
#include "tla/tensor.hpp"

namespace CubeSelf::Gemm::Tile {

template<
    class ArchTag,
    /// GemmType for matrix operand
    class GmType,
    class L1Type=void
    >
struct CopyGmToL1 {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to l1, can not find the specialization.");
};

template <
    class ArchTag,
    /// GemmType for matrix operand
    class GmType,
    class L1Type = void
>
struct CopyGmToL1IntervalDataCopy {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to l1, can not find the specialization.");
};


////////////////////////////////////////
/// Using the standard strided DataCopy interface to implement nd2nz
/// transfer may achieve higher data transfer efficiency when the data block shape is short and wide
/// Partial specialization for AtlasA2, half, RowMajor in and zN out.
template<>
struct CopyGmToL1IntervalDataCopy<Catlass::Arch::AtlasA2, Catlass::Gemm::GemmType<half,Catlass::layout::RowMajor>>
{
    using LayoutDst = Catlass::layout::zN;
    using LayoutSrc = Catlass::layout::RowMajor;
    using Element = half;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);
    // Catlass::BYTE_PER_C0 -> 32 Byte， 一个datablock的size

    // Methods

    CATLASS_DEVICE
    CopyGmToL1IntervalDataCopy(){};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        for(int i=0; i < layoutSrc.shape(0);++i)
        {
            // row_major: shape{row,col}
            /*
            Index rowsInFractal = 0,           /// Number of rows inside the fractal
            Index rowsByFractal = 0,           /// number of rows by the fractal
            Index colsInFractal = 0,           /// number of cols inside the fractal
            Index colsByFractal = 0,           /// number of cols by the fractal
            */
            // zN: shape_(MakeCoord(rowsInFractal, rowsByFractal, colsInFractal, colsByFractal))
            AscendC::DataCopyParams  dataCopyParams(
                CeilDiv(layoutSrc.shape(1),layoutDst.shape(2)),
                layoutDst.shape(2) / ELE_NUM_PER_C0,
                0,
                (layoutDst.stride(3) - layoutDst.shape(2)) / ELE_NUM_PER_C0
            );

            /*
            row_major: stride: {cols,1}

            LongIndex strideRowsInFractal = 0, /// number of elements between adjacent rows inside the fractal
            LongIndex strideRowsByFractal = 0, /// number of elements between adjacent fractal rows
            LongIndex strideColsInFractal = 0, /// number of elements between adjacent cols inside the fractal
            LongIndex strideColsByFractal = 0) /// number of elements between adjacent fractal cols
            zN: stride: {strideRowsInFractal, strideRowsByFractal, strideColsInFractal, strideColsByFractal}
            */

            AscendC::DataCopy(dstTensor[i * layoutDst.shape(2)], srcTensor[i * layoutSrc.stride(0)], dataCopyParams);
        }    
    }  
};

/// Partial specialization for AtlasA2, half, PaddingRowMajor in and zN out.
/// Using the standard strided DataCopy interface to implement nd2nz
/// transfer may achieve higher data transfer efficiency when the data block shape is short and wide
template<>
struct CopyGmToL1IntervalDataCopy<Catlass::Arch::AtlasA2, Catlass::Gemm::GemmType<half,Catlass::layout::PaddingRowMajor>>{

    using LayoutDst = Catlass::layout::zN;
    using LayoutSrc = Catlass::layout::PaddingRowMajor;
    using Element = half;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    CATLASS_DEVICE
    CopyGmToL1IntervalDataCopy(){};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    ){
        for(int i=0;i<layoutSrc.orgShape(0);++i){
            AscendC::DataCopyParams  dataCopyParams(
                CeilDiv(layoutSrc.orgShape(1), layoutDst.shape(2)),
                layoutDst.shape(2) / ELE_NUM_PER_C0,
                0,
                (layoutDst.stride(3) - layoutDst.shape(2)) / ELE_NUM_PER_C0
            );

            AscendC::DataCopy(dstTensor[i * layoutDst.shape(2)], srcTensor[i * layoutSrc.stride(0)], dataCopyParams);
        }
    }
};

/// Partial specialization for AtlasA2, half, ColumnMajor in and zN out.
/// Using the standard strided DataCopy interface to implement nd2nz
/// transfer may achieve higher data transfer efficiency when the data block shape is tall and narrow
template<>
struct CopyGmToL1IntervalDataCopy<Catlass::Arch::AtlasA2, Catlass::Gemm::GemmType<half,Catlass::layout::ColumnMajor>>{
    using LayoutDst = Catlass::layout::nZ;
    using LayoutSrc = Catlass::layout::ColumnMajor;

    using Element = half;
    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);

    //Methods

    CATLASS_DEVICE
    CopyGmToL1IntervalDataCopy() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    ){
        for(int i=0; i < layoutSrc.shape(1); ++i){
            AscendC::DataCopyParams dataCopyParams(
                CeilDiv(layoutSrc.shape(0),layoutDst.shape(0)),
                layoutDst.shape(0) / ELE_NUM_PER_C0,
                0,
                (layoutDst.stride(1) - layoutDst.shape(0))/ELE_NUM_PER_C0
            );

            AscendC::DataCopy(dstTensor[i * layoutDst.shape(0)], 
                srcTensor[i * layoutSrc.stride(1)], dataCopyParams);
        }
    }
};

/// Partial specialization for AtlasA2, half, PaddingColumnMajor in and zN out.
/// Using the standard strided DataCopy interface to implement nd2nz
/// transfer may achieve higher data transfer efficiency when the data block shape is tall and narrow
template<>
struct CopyGmToL1IntervalDataCopy<Catlass::Arch::AtlasA2, Catlass::Gemm::GemmType<half,Catlass::layout::PaddingColumnMajor>>
{
    using LayoutDst = Catlass::layout::nZ;
    using LayoutSrc = Catlass::layout::PaddingColumnMajor;
    using Element = half;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);

    //Methods

    CATLASS_DEVICE
    CopyGmToL1IntervalDataCopy() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    ){
        for(int i=0; i < layoutSrc.orgShape(1); ++i){
            AscendC::DataCopyParams dataCopyParams(
                CeilDiv(layoutSrc.orgShape(0),layoutDst.shape(0)),
                layoutDst.shape(0) / ELE_NUM_PER_C0,
                0,
                (layoutDst.stride(1) - layoutDst.shape(0))/ELE_NUM_PER_C0
            );

            AscendC::DataCopy(dstTensor[i * layoutDst.shape(0)], 
                srcTensor[i * layoutSrc.stride(1)], dataCopyParams);
        }
    }
};

/// new add gemm
template<class ArchTag, class Element>
struct CopyGmToL1<ArchTag, 
    Catlass::Gemm::GemmType<Element,Catlass::layout::RowMajor>,
    Catlass::Gemm::GemmType<Element,Catlass::layout::zN,AscendC::TPosition::A1>>
{
    using LayoutDst = Catlass::layout::zN;
    using LayoutSrc = Catlass::layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(1);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (layoutSrc.stride(0) < Catlass::STRIDE_LIMIT) {
            intriParams.nValue = layoutSrc.shape(0);
            intriParams.srcDValue = layoutSrc.stride(0);
            intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.nValue = 1;
            intriParams.srcDValue = 0;
            intriParams.dstNzNStride = 0;
            for (uint32_t i = 0; i < layoutSrc.shape(0); i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(0)], intriParams);
            }
        }
    }

};

template<class ArchTag, class Element>
struct CopyGmToL1<ArchTag,
    Catlass::Gemm::GemmType<Element,Catlass::layout::RowMajor>,
    Catlass::Gemm::GemmType<Element,Catlass::layout::zZ,AscendC::TPosition::B1>>
{
    using LayoutDst = Catlass::layout::zZ;
    using LayoutSrc = Catlass::layout::RowMajor;
    
    static const uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);

     // Mehtods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;
        uint32_t srcNdStride = Catlass::C0_NUM_PER_FRACTAL * layoutSrc.stride(0);
        uint32_t ndNum = layoutSrc.shape(0) / Catlass::C0_NUM_PER_FRACTAL;
        uint32_t remains = layoutSrc.shape(0) % Catlass::C0_NUM_PER_FRACTAL;
        if (srcNdStride < Catlass::STRIDE_LIMIT) {
            if (ndNum) {
                intriParams.ndNum = ndNum;
                intriParams.nValue = Catlass::C0_NUM_PER_FRACTAL;
                intriParams.dValue = layoutSrc.shape(1);
                intriParams.srcNdMatrixStride = srcNdStride;
                intriParams.srcDValue = layoutSrc.stride(0);

                intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
                intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;

                intriParams.dstNzMatrixStride = layoutDst.stride(1);

                AscendC::DataCopy(dstTensor, srcTensor, intriParams);
            }

            if (remains) {
                AscendC::Nd2NzParams tailParams;
                tailParams.ndNum = 1;
                tailParams.nValue = remains;
                tailParams.dValue = layoutSrc.shape(1);
                tailParams.srcNdMatrixStride = srcNdStride;
                tailParams.srcDValue = layoutSrc.stride(0);

                tailParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
                tailParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
                tailParams.dstNzMatrixStride = 0;  //`

                AscendC::DataCopy(dstTensor[ndNum * layoutDst.stride(1)], srcTensor[ndNum * srcNdStride], tailParams);
            }
        } else if (layoutSrc.stride(0) < Catlass::STRIDE_LIMIT) {
            for (uint32_t i = 0; i < ndNum; i++) {
                AscendC::Nd2NzParams intriParams;
                intriParams.ndNum = 1;
                intriParams.nValue = Catlass::C0_NUM_PER_FRACTAL;
                intriParams.dValue = layoutSrc.shape(1);
                intriParams.srcNdMatrixStride = 0;
                intriParams.srcDValue = layoutSrc.stride(0);

                intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
                intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
                intriParams.dstNzMatrixStride = 0;

                AscendC::DataCopy(dstTensor[i * layoutDst.stride(1)], srcTensor[i * srcNdStride], intriParams);
            }
            if (remains) {
                AscendC::Nd2NzParams tailParams;
                tailParams.ndNum = 1;
                tailParams.nValue = remains;
                tailParams.dValue = layoutSrc.shape(1);
                tailParams.srcNdMatrixStride = 0;
                tailParams.srcDValue = layoutSrc.stride(0);

                tailParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
                tailParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
                tailParams.dstNzMatrixStride = 0;

                AscendC::DataCopy(dstTensor[ndNum * layoutDst.stride(1)], srcTensor[ndNum * srcNdStride], tailParams);
            }
        } else {
            for (uint32_t i = 0; i < layoutSrc.shape(0); i++) {
                uint32_t idxR0 = i / Catlass::C0_NUM_PER_FRACTAL;
                uint32_t idxInR0 = i % Catlass::C0_NUM_PER_FRACTAL;

                AscendC::Nd2NzParams intriParams;
                intriParams.ndNum = 1;
                intriParams.nValue = 1;
                intriParams.dValue = layoutSrc.shape(1);
                intriParams.srcNdMatrixStride = 0;
                intriParams.srcDValue = 0;

                intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
                intriParams.dstNzNStride = 0;
                intriParams.dstNzMatrixStride = 0;

                uint32_t offsetDst = i * idxR0 * layoutDst.stride(1) + idxInR0 * ELE_NUM_PER_C0;
                uint32_t offsetSrc = i * layoutSrc.stride(0);
                AscendC::DataCopy(dstTensor[offsetDst], srcTensor[offsetSrc], intriParams);
            }
        }
    }
};

template<class ArchTag, class Element>
struct CopyGmToL1<ArchTag,
    Catlass::Gemm::GemmType<Element, Catlass::layout::ColumnMajor>,
    Catlass::Gemm::GemmType<Element,Catlass::layout::nN,AscendC::TPosition::A1>>
{

    using LayoutDst = Catlass::layout::nN;
    using LayoutSrc = Catlass::layout::ColumnMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;
        uint32_t srcNdStride = Catlass::C0_NUM_PER_FRACTAL * layoutSrc.stride(1);
        uint32_t ndNum = layoutSrc.shape(1) / Catlass::C0_NUM_PER_FRACTAL;
        uint32_t remains = layoutSrc.shape(1) % Catlass::C0_NUM_PER_FRACTAL;
        if (srcNdStride < Catlass::STRIDE_LIMIT) {
            if (ndNum) {
                intriParams.ndNum = ndNum;
                intriParams.nValue = Catlass::C0_NUM_PER_FRACTAL;
                intriParams.dValue = layoutSrc.shape(0);
                intriParams.srcNdMatrixStride = srcNdStride;
                intriParams.srcDValue = layoutSrc.stride(1);

                intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;

                intriParams.dstNzMatrixStride = layoutDst.stride(3);

                AscendC::DataCopy(dstTensor, srcTensor, intriParams);
            }

            if (remains) {
                AscendC::Nd2NzParams tailParams;
                tailParams.ndNum = 1;
                tailParams.nValue = remains;
                tailParams.dValue = layoutSrc.shape(0);
                tailParams.srcNdMatrixStride = srcNdStride;
                tailParams.srcDValue = layoutSrc.stride(1);

                tailParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                tailParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
                tailParams.dstNzMatrixStride = 0;

                AscendC::DataCopy(dstTensor[ndNum * layoutDst.stride(3)], srcTensor[ndNum * srcNdStride], tailParams);
            }
        } else if (layoutSrc.stride(1) < Catlass::STRIDE_LIMIT) {
            for (uint32_t i = 0; i < ndNum; i++) {
                AscendC::Nd2NzParams intriParams;
                intriParams.ndNum = 1;
                intriParams.nValue = Catlass::C0_NUM_PER_FRACTAL;
                intriParams.dValue = layoutSrc.shape(0);
                intriParams.srcNdMatrixStride = 0;
                intriParams.srcDValue = layoutSrc.stride(1);

                intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
                intriParams.dstNzMatrixStride = 0;

                AscendC::DataCopy(dstTensor[i * layoutDst.stride(3)], srcTensor[i * srcNdStride], intriParams);
            }
            if (remains) {
                AscendC::Nd2NzParams tailParams;
                tailParams.ndNum = 1;
                tailParams.nValue = remains;
                tailParams.dValue = layoutSrc.shape(0);
                tailParams.srcNdMatrixStride = 0;
                tailParams.srcDValue = layoutSrc.stride(1);

                tailParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                tailParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
                tailParams.dstNzMatrixStride = 0;

                AscendC::DataCopy(dstTensor[ndNum * layoutDst.stride(3)], srcTensor[ndNum * srcNdStride], tailParams);
            }
        } else {
            for (uint32_t i = 0; i < layoutSrc.shape(1); i++) {
                uint32_t idxR0 = i / Catlass::C0_NUM_PER_FRACTAL;
                uint32_t idxInR0 = i % Catlass::C0_NUM_PER_FRACTAL;

                AscendC::Nd2NzParams intriParams;
                intriParams.ndNum = 1;
                intriParams.nValue = 1;
                intriParams.dValue = layoutSrc.shape(0);
                intriParams.srcNdMatrixStride = 0;
                intriParams.srcDValue = 0;

                intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                intriParams.dstNzNStride = 0;
                intriParams.dstNzMatrixStride = 0;

                uint32_t offsetDst = i * idxR0 * layoutDst.stride(3) + idxInR0 * ELE_NUM_PER_C0;
                uint32_t offsetSrc = i * layoutSrc.stride(1);
                AscendC::DataCopy(dstTensor[offsetDst], srcTensor[offsetSrc], intriParams);
            }
        }
    }
};

template<class ArchTag, class Element>
struct CopyGmToL1<ArchTag,
    Catlass::Gemm::GemmType<Element,Catlass::layout::ColumnMajor>,
    Catlass::Gemm::GemmType<Element,Catlass::layout::nZ,AscendC::TPosition::B1>>
{
    using LayoutDst = Catlass::layout::nZ;
    using LayoutSrc = Catlass::layout::ColumnMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(0);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (layoutSrc.stride(1) < Catlass::STRIDE_LIMIT) {
            intriParams.nValue = layoutSrc.shape(1);
            intriParams.srcDValue = layoutSrc.stride(1);
            intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.nValue = 1;
            intriParams.srcDValue = 0;
            intriParams.dstNzNStride = 0;
            for (uint32_t i = 0; i < layoutSrc.shape(1); i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(1)], intriParams);
            }
        }
    }
};

template<class ArchTag, class Element>
struct CopyGmToL1<ArchTag, 
    Catlass::Gemm::GemmType<Element,Catlass::layout::ColumnMajor>, 
    Catlass::Gemm::GemmType<Element,Catlass::layout::nZ,AscendC::TPosition::A1>>
{
    using LayoutDst = Catlass::layout::nZ;
    using LayoutSrc = Catlass::layout::ColumnMajor;

    static const uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);

    // Methods

    CATLASS_DEVICE
    CopyGmToL1(){};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(0);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (layoutSrc.stride(1) < Catlass::STRIDE_LIMIT) {
            intriParams.nValue = layoutSrc.shape(1);
            intriParams.srcDValue = layoutSrc.stride(1);
            intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.nValue = 1;
            intriParams.srcDValue = 0;
            intriParams.dstNzNStride = 0;
            for (uint32_t i = 0; i < layoutSrc.shape(1); i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(1)], intriParams);
            }
        }
    }
};

////////////////////////////////////////

///////////////////////////////////////
/// new add gemv, VectorLayout -> zN
template <class ArchTag, class Element>
struct CopyGmToL1<ArchTag, Catlass::Gemm::GemmType<Element, Catlass::layout::VectorLayout>, Catlass::Gemm::GemmType<Element, Catlass::layout::zN, AscendC::TPosition::A1>> {
    using LayoutDst = Catlass::layout::zN;
    using LayoutSrc = Catlass::layout::VectorLayout;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);

    // Methods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(0);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;
        intriParams.nValue = 1;
        intriParams.srcDValue = layoutSrc.shape(0);
        intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
        AscendC::DataCopy(dstTensor, srcTensor, intriParams);
    }
};

template<class ArchTag, class Element>
struct CopyGmToL1<ArchTag, 
                  Catlass::Gemm::GemmType<Element, Catlass::layout::ColumnMajor>, 
                  Catlass::Gemm::GemmType<Element, Catlass::layout::nN, 
                  AscendC::TPosition::B1>>
{
    using LayoutDst = Catlass::layout::nN;
    using LayoutSrc = Catlass::layout::ColumnMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);

    // Methods

    CATLASS_DEVICE
    CopyGmToL1();

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;
        uint32_t srcNdStride = Catlass::C0_NUM_PER_FRACTAL * layoutSrc.stride(1);
        uint32_t ndNum = layoutSrc.shape(1) / Catlass::C0_NUM_PER_FRACTAL;
        uint32_t remains = layoutSrc.shape(1) % Catlass::C0_NUM_PER_FRACTAL;
        if (srcNdStride < Catlass::STRIDE_LIMIT) {
            if (ndNum) {
                intriParams.ndNum = ndNum;
                intriParams.nValue = Catlass::C0_NUM_PER_FRACTAL;
                intriParams.dValue = layoutSrc.shape(0);
                intriParams.srcNdMatrixStride = srcNdStride;
                intriParams.srcDValue = layoutSrc.stride(1);

                intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;

                intriParams.dstNzMatrixStride = layoutDst.stride(3);

                AscendC::DataCopy(dstTensor, srcTensor, intriParams);
            }

            if (remains) {
                AscendC::Nd2NzParams tailParams;
                tailParams.ndNum = 1;
                tailParams.nValue = remains;
                tailParams.dValue = layoutSrc.shape(0);
                tailParams.srcNdMatrixStride = srcNdStride;
                tailParams.srcDValue = layoutSrc.stride(1);

                tailParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                tailParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
                tailParams.dstNzMatrixStride = 0;

                AscendC::DataCopy(dstTensor[ndNum * layoutDst.stride(3)], srcTensor[ndNum * srcNdStride], tailParams);
            }
        } else if (layoutSrc.stride(1) < Catlass::STRIDE_LIMIT) {
            for (uint32_t i = 0; i < ndNum; i++) {
                AscendC::Nd2NzParams intriParams;
                intriParams.ndNum = 1;
                intriParams.nValue = Catlass::C0_NUM_PER_FRACTAL;
                intriParams.dValue = layoutSrc.shape(0);
                intriParams.srcNdMatrixStride = 0;
                intriParams.srcDValue = layoutSrc.stride(1);

                intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
                intriParams.dstNzMatrixStride = 0;

                AscendC::DataCopy(dstTensor[i * layoutDst.stride(3)], srcTensor[i * srcNdStride], intriParams);
            }
            if (remains) {
                AscendC::Nd2NzParams tailParams;
                tailParams.ndNum = 1;
                tailParams.nValue = remains;
                tailParams.dValue = layoutSrc.shape(0);
                tailParams.srcNdMatrixStride = 0;
                tailParams.srcDValue = layoutSrc.stride(1);

                tailParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                tailParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
                tailParams.dstNzMatrixStride = 0;

                AscendC::DataCopy(dstTensor[ndNum * layoutDst.stride(3)], srcTensor[ndNum * srcNdStride], tailParams);
            }
        } else {
            for (uint32_t i = 0; i < layoutSrc.shape(1); i++) {
                uint32_t idxR0 = i / Catlass::C0_NUM_PER_FRACTAL;
                uint32_t idxInR0 = i % Catlass::C0_NUM_PER_FRACTAL;

                AscendC::Nd2NzParams intriParams;
                intriParams.ndNum = 1;
                intriParams.nValue = 1;
                intriParams.dValue = layoutSrc.shape(0);
                intriParams.srcNdMatrixStride = 0;
                intriParams.srcDValue = 0;

                intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                intriParams.dstNzNStride = 0;
                intriParams.dstNzMatrixStride = 0;

                uint32_t offsetDst = i * idxR0 * layoutDst.stride(3) + idxInR0 * ELE_NUM_PER_C0;
                uint32_t offsetSrc = i * layoutSrc.stride(1);
                AscendC::DataCopy(dstTensor[offsetDst], srcTensor[offsetSrc], intriParams);
            }
        }
    }
};

template <class ArchTag, class Element>
struct CopyGmToL1<ArchTag,
        Catlass::Gemm::GemmType<Element,Catlass::layout::RowMajor>,
        Catlass::Gemm::GemmType<Element,Catlass::layout::zN,AscendC::TPosition::B1>>
{
    using LayoutDst = Catlass::layout::zN;
    using LayoutSrc = Catlass::layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);

    // Methods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(1);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (layoutSrc.stride(0) < Catlass::STRIDE_LIMIT) {
            intriParams.nValue = layoutSrc.shape(0);
            intriParams.srcDValue = layoutSrc.stride(0);
            intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.nValue = 1;
            intriParams.srcDValue = 0;
            intriParams.dstNzNStride = 0;
            for (uint32_t i = 0; i < layoutSrc.shape(0); i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(0)], intriParams);
            }
        }
    }
};

/////////////////////////////////

/// Partial specialization for AtlasA2, RowMajor in and zN out.
template<class Element>
struct CopyGmToL1<Catlass::Arch::AtlasA2,
    Catlass::Gemm::GemmType<Element,Catlass::layout::RowMajor>>
{
    using LayoutDst = Catlass::layout::zN;
    using LayoutSrc = Catlass::layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);

    // Methods

    CATLASS_DEVICE
    CopyGmToL1(){};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(1);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (layoutSrc.stride(0) < Catlass::STRIDE_LIMIT) {
            intriParams.nValue = layoutSrc.shape(0);
            intriParams.srcDValue = layoutSrc.stride(0);
            intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.nValue = 1;
            intriParams.srcDValue = 0;
            intriParams.dstNzNStride = 0;
            for (uint32_t i = 0; i < layoutSrc.shape(0); i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(0)], intriParams);
            }
        }
    }

    // layoutSrc must be the layout of one of the src matrices
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        uint32_t ndNum, uint32_t srcNdMatrixStride,
        uint32_t dstNzNStride, uint32_t dstNzMatrixStride,
        uint32_t dstNzC0Stride)
    { // 支持多个ND矩阵的版本
        AscendC::Nd2NzParams intriParams;

        intriParams.nValue = layoutSrc.shape(0);
        intriParams.dValue = layoutSrc.shape(1);
        intriParams.srcDValue = layoutSrc.stride(0);
        intriParams.dstNzNStride = dstNzNStride;
        intriParams.dstNzC0Stride = dstNzC0Stride;
        if (srcNdMatrixStride < Catlass::STRIDE_LIMIT) {
            intriParams.ndNum = ndNum;
            intriParams.srcNdMatrixStride = srcNdMatrixStride;
            intriParams.dstNzMatrixStride = dstNzMatrixStride;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.ndNum = 1;
            intriParams.srcNdMatrixStride = 0;
            intriParams.dstNzMatrixStride = 0;
            for (uint32_t i = 0; i < ndNum; i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * srcNdMatrixStride], intriParams);
            }
        }
    }
};

/// Partial specialization for AtlasA2, ColumnMajor in and nZ out.
template<class Element>
struct CopyGmToL1<Catlass::Arch::AtlasA2, Catlass::Gemm::GemmType<Element, Catlass::layout::ColumnMajor>>
{
    using LayoutDst = Catlass::layout::nZ;
    using LayoutSrc = Catlass::layout::ColumnMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(0);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (layoutSrc.stride(1) < Catlass::STRIDE_LIMIT) {
            intriParams.nValue = layoutSrc.shape(1);
            intriParams.srcDValue = layoutSrc.stride(1);
            intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.nValue = 1;
            intriParams.srcDValue = 0;
            intriParams.dstNzNStride = 0;
            for (uint32_t i = 0; i < layoutSrc.shape(1); i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(1)], intriParams);
            }
        }
    }
};

/// Partial specialization for zN in and zN out.
template<class ArchTag, class Element>
struct CopyGmToL1<ArchTag, Catlass::Gemm::GemmType<Element,Catlass::layout::zN>> 
{
    using LayoutDst = Catlass::layout::zN;
    using LayoutSrc = Catlass::layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);

     // Mehtods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const & dstTensor,
        AscendC::GlobalTensor<Element> const & srcTensor,
        LayoutDst const & layoutDst,
        LayoutSrc const & layoutSrc)
    {
        /*
        zN: Index orgRows = 0,                 /// Number of rows of origin matrices
            Index orgCols = 0,                 /// Number of cols of origin matrices
            orgShape(orgRows,orgCols)
        */
        uint32_t blockCount = CeilDiv<ELE_NUM_PER_C0>(layoutSrc.orgShape(1));
        uint32_t blockLen = RoundUp<Catlass::C0_NUM_PER_FRACTAL>(layoutSrc.orgShape(0));

        AscendC::DataCopyParams repeatParams;

        if (layoutSrc.stride(3) / ELE_NUM_PER_C0 < Catlass::STRIDE_LIMIT) {
            repeatParams.blockCount = blockCount;
            repeatParams.blockLen = blockLen;
            repeatParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_C0 - blockLen;
            repeatParams.dstStride = layoutDst.stride(3) / ELE_NUM_PER_C0 - blockLen;
            AscendC::DataCopy(dstTensor, srcTensor, repeatParams);
        } else {
            repeatParams.blockCount = 1;
            repeatParams.blockLen = blockLen;
            repeatParams.srcStride = 0;
            repeatParams.dstStride = 0;
            for (uint32_t i = 0; i < blockCount; i++) {
                uint64_t dstOffset = i * layoutDst.stride(3);
                uint64_t srcOffset = i * layoutSrc.stride(3);
                AscendC::DataCopy(dstTensor[dstOffset], srcTensor[srcOffset], repeatParams);
            }
        }
    }
};

/// Partial specialization for nZ in and nZ out.
template<class ArchTag, class Element>
struct CopyGmToL1<ArchTag, Catlass::Gemm::GemmType<Element,Catlass::layout::nZ>>
{
    using LayoutDst = Catlass::layout::nZ;
    using LayoutSrc = Catlass::layout::nZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);
    
    // Methods

    CATLASS_DEVICE
    CopyGmToL1(){};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        uint32_t blockCount = CeilDiv<ELE_NUM_PER_C0>(layoutSrc.orgShape(0));
        uint32_t blockLen = RoundUp<Catlass::C0_NUM_PER_FRACTAL>(layoutSrc.orgShape(1));

        AscendC::DataCopyParams repeatParams;

        if (layoutSrc.stride(1) / ELE_NUM_PER_C0 < Catlass::STRIDE_LIMIT) {
            repeatParams.blockCount = blockCount;
            repeatParams.blockLen = blockLen;
            repeatParams.srcStride = layoutSrc.stride(1) / ELE_NUM_PER_C0 - blockLen;
            repeatParams.dstStride = layoutDst.stride(1) / ELE_NUM_PER_C0 - blockLen;
            AscendC::DataCopy(dstTensor, srcTensor, repeatParams);
        } else {
            repeatParams.blockCount = 1;
            repeatParams.blockLen = blockLen;
            repeatParams.srcStride = 0;
            repeatParams.dstStride = 0;
            for (uint32_t i = 0; i < blockCount; i++) {
                uint64_t dstOffset = i * layoutDst.stride(1);
                uint64_t srcOffset = i * layoutSrc.stride(1);
                AscendC::DataCopy(dstTensor[dstOffset], srcTensor[srcOffset], repeatParams);
            }
        }
    }
};

/// Partial specialization for AtlasA2, PaddingRowMajor in and zN out.
template<class Element>
struct CopyGmToL1<Catlass::Arch::AtlasA2, 
    Catlass::Gemm::GemmType<Element, Catlass::layout::PaddingRowMajor>>
{
    using LayoutDst = Catlass::layout::zN;
    using LayoutSrc = Catlass::layout::PaddingRowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);

    // Methods

    CATLASS_DEVICE
    CopyGmToL1(){};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.orgShape(1);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        intriParams.nValue = layoutSrc.orgShape(0);
        intriParams.srcDValue = layoutSrc.stride(0);
        intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
        AscendC::DataCopy(dstTensor, srcTensor, intriParams);
    }

};

/// Partial specialization for AtlasA2, ColumnMajor in and nZ out.
template<class Element>
struct CopyGmToL1<Catlass::Arch::AtlasA2, Catlass::Gemm::GemmType<Element,Catlass::layout::PaddingColumnMajor>>
{
    using LayoutDst = Catlass::layout::nZ;
    using LayoutSrc = Catlass::layout::PaddingColumnMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);

    CATLASS_DEVICE
    CopyGmToL1(){};

   CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.orgShape(0);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        intriParams.nValue = layoutSrc.orgShape(1);
        intriParams.srcDValue = layoutSrc.stride(2);
        intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
        AscendC::DataCopy(dstTensor, srcTensor, intriParams);
    } 
};

/// Partial specialization for AtlasA2, RowMajor in and RowMajor out.
template<class Element>
struct CopyGmToL1<Catlass::Arch::AtlasA2, 
    Catlass::Gemm::GemmType<Element,Catlass::layout::RowMajor>,
    Catlass::Gemm::GemmType<Element,Catlass::layout::RowMajor,AscendC::TPosition::A1>>
{
    using LayoutDst = Catlass::layout::RowMajor;
    using LayoutSrc = Catlass::layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_BLK = Catlass::BYTE_PER_BLK / sizeof(Element);
    static constexpr uint32_t BLOCK_LEN_LIMIT = 65536;
    static constexpr uint32_t MAX_REPEAT = 4095;

    // Methods

    CATLASS_DEVICE
    CopyGmToL1(){};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        uint32_t rows = layoutSrc.shape(0);
        uint32_t cols = layoutSrc.shape(1);
        uint32_t srcStride = (layoutSrc.stride(0) - layoutSrc.shape(1)) / ELE_NUM_PER_BLK;
        uint32_t dstStride = (layoutDst.stride(0) - layoutDst.shape(1)) / ELE_NUM_PER_BLK;

        if ((layoutSrc.shape(1) == layoutSrc.stride(0)) && (layoutDst.shape(1) == layoutDst.stride(0))) {
            AscendC::DataCopy(dstTensor, srcTensor, rows * cols);
        } else if (srcStride < Catlass::STRIDE_LIMIT && dstStride < Catlass::STRIDE_LIMIT && (cols / ELE_NUM_PER_BLK) < BLOCK_LEN_LIMIT) {
            uint32_t rLoops = CeilDiv(rows, MAX_REPEAT);
            for (uint32_t i = 0; i < rLoops; ++i) {
                uint32_t rActual = (i < rLoops - 1) ? MAX_REPEAT : rows - i * MAX_REPEAT;
                AscendC::DataCopyParams dataCopyParams(
                    rActual, cols / ELE_NUM_PER_BLK, srcStride, dstStride
                );
                AscendC::DataCopy(dstTensor[i * MAX_REPEAT * layoutDst.stride(0)],
                         srcTensor[i * MAX_REPEAT * layoutSrc.stride(0)], dataCopyParams);
            }
        } else {
            for (uint32_t i = 0; i < rows; ++i) {
                AscendC::DataCopy(dstTensor[i * layoutDst.stride(0)], srcTensor[i * layoutSrc.stride(0)], cols);
            }
        }
    }
};

///////////////////////////////////////////TileCopyTla//////////////////////////////////////////////////////
/// Partial specialization for CopyGmToL1, AtlasA2, RowMajor in and zN out.
template<class ElementSrc,class ElementDst, class LayoutSrc_, class LayoutDst_>
struct TileCopyTla<Catlass::Arch::AtlasA2, 
    tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc_, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst_, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc_>::value &&
                     tla::detail::iszN<ElementDst, LayoutDst_>::value>> {
    using LayoutDst = LayoutDst_;
    using LayoutSrc = LayoutSrc_;
    using TensorDst = tla::Tensor<AscendC::LocalTensor<ElementDst>, 
        LayoutDst, AscendC::TPosition::A1>;
    using TensorSrc = tla::Tensor<AscendC::GlobalTensor<ElementSrc>, 
        LayoutSrc, AscendC::TPosition::GM>;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(ElementSrc);

    // Mehtods

    CATLASS_DEVICE
    TileCopyTla() {};

    CATLASS_DEVICE
    void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        const uint32_t nValue = tla::get<0>(srcTensor.shape());
        const uint32_t dValue = tla::get<1>(srcTensor.shape());
        const uint32_t srcDValue = tla::get<0>(srcTensor.stride());
        const uint32_t dstInnerStrideRow = tla::get<0, 0>(dstTensor.stride());
        const uint32_t dstOuterStrideCol = tla::get<1, 1>(dstTensor.stride());

        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = dValue;
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = dstOuterStrideCol / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (srcDValue < Catlass::STRIDE_LIMIT) {
            intriParams.nValue = nValue;
            intriParams.srcDValue = srcDValue;
            intriParams.dstNzNStride = dstInnerStrideRow / ELE_NUM_PER_C0;
            AscendC::DataCopy(dstTensor.data(), srcTensor.data(), intriParams);
        } else {
            intriParams.nValue = 1;
            intriParams.srcDValue = 0;
            intriParams.dstNzNStride = 0;
            for (uint32_t i = 0; i < nValue; i++) {
                AscendC::DataCopy(dstTensor.data()[i * ELE_NUM_PER_C0], srcTensor.data()[i * srcDValue], intriParams);
            }
        }
    }
};

/// Partial specialization for CopyGmToL1, AtlasA2, ColumnMajor in and nZ out.
template <class ElementSrc, class ElementDst, class LayoutSrc_, class LayoutDst_>
struct TileCopyTla<Catlass::Arch::AtlasA2, 
    tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc_, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst_, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isColumnMajor<LayoutSrc_>::value &&
                     tla::detail::isnZ<ElementDst, LayoutDst_>::value>> {
    using LayoutDst = LayoutDst_;
    using LayoutSrc = LayoutSrc_;
    using TensorDst = tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, AscendC::TPosition::A1>;
    using TensorSrc = tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, AscendC::TPosition::GM>;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(ElementSrc);

    // Mehtods

    CATLASS_DEVICE
    TileCopyTla() {};

    CATLASS_DEVICE
    void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        const uint32_t nValue = tla::get<1>(srcTensor.shape());
        const uint32_t dValue = tla::get<0>(srcTensor.shape());
        const uint32_t srcDValue = tla::get<1>(srcTensor.stride());
        const uint32_t dstInnerStrideRow = tla::get<1, 0>(dstTensor.stride());
        const uint32_t dstOuterStrideCol = tla::get<0, 1>(dstTensor.stride());

        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = dValue;
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = dstOuterStrideCol / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (srcDValue < Catlass::STRIDE_LIMIT) {
            intriParams.nValue = nValue;
            intriParams.srcDValue = srcDValue;
            intriParams.dstNzNStride = dstInnerStrideRow / ELE_NUM_PER_C0;
            AscendC::DataCopy(dstTensor.data(), srcTensor.data(), intriParams);
        } else {
            intriParams.nValue = 1;
            intriParams.srcDValue = 0;
            intriParams.dstNzNStride = 0;
            for (uint32_t i = 0; i < nValue; i++) {
                AscendC::DataCopy(dstTensor.data()[i * ELE_NUM_PER_C0], srcTensor.data()[i * srcDValue], intriParams);
            }
        }
    }
};

/// Partial specialization for TileCopyTlaExt, CopyGmToL1, AtlasA2, PaddingRowMajor in and zN out.
template <class ElementSrc, class ElementDst, class LayoutSrc_, class LayoutDst_>
struct TileCopyTlaExt<Catlass::Arch::AtlasA2,
    tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc_, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst_, AscendC::TPosition::A1>,
    Catlass::layout::RowMajor, Catlass::layout::zN> {
    using LayoutDst = LayoutDst_;
    using LayoutSrc = LayoutSrc_;
    using TensorDst = tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, AscendC::TPosition::A1>;
    using TensorSrc = tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, AscendC::TPosition::GM>;
    using ActualShape = tla::Shape<uint32_t, uint32_t>;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(ElementSrc);

    // Mehtods

    CATLASS_DEVICE
    TileCopyTlaExt() {};

    CATLASS_DEVICE
    void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, ActualShape actualShape)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = tla::get<1>(actualShape);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = tla::get<1, 1>(dstTensor.stride()) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        intriParams.nValue = tla::get<0>(actualShape);
        intriParams.srcDValue = tla::get<0>(srcTensor.stride());
        intriParams.dstNzNStride = tla::get<0, 0>(dstTensor.stride()) / ELE_NUM_PER_C0;
        AscendC::DataCopy(dstTensor.data(), srcTensor.data(), intriParams);
    }
};

/// Partial specialization for TileCopyTlaExt, CopyGmToL1, AtlasA2, PaddingRowMajor in and zN out.
template <class ElementSrc, class ElementDst, class LayoutSrc_, class LayoutDst_>
struct TileCopyTlaExt<Catlass::Arch::AtlasA2, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc_, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst_, AscendC::TPosition::A1>,
    Catlass::layout::PaddingRowMajor, Catlass::layout::zN> {
    using LayoutDst = LayoutDst_;
    using LayoutSrc = LayoutSrc_;
    using TensorDst = tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, AscendC::TPosition::A1>;
    using TensorSrc = tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, AscendC::TPosition::GM>;
    using ActualShape = tla::Shape<uint32_t, uint32_t>;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(ElementSrc);

    // Mehtods

    CATLASS_DEVICE
    TileCopyTlaExt() {};

    CATLASS_DEVICE
    void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, ActualShape actualShape)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = tla::get<1>(actualShape);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = tla::get<1, 1>(dstTensor.stride()) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        intriParams.nValue = tla::get<0>(actualShape);
        intriParams.srcDValue = tla::get<0, 0>(srcTensor.stride());
        intriParams.dstNzNStride = tla::get<0, 0>(dstTensor.stride()) / ELE_NUM_PER_C0;
        AscendC::DataCopy(dstTensor.data(), srcTensor.data(), intriParams);
    }
};

/// Partial specialization for TileCopyTlaExt, CopyGmToL1, AtlasA2, PaddingColumnMajor in and nZ out.
template <class ElementSrc, class ElementDst, class LayoutSrc_, class LayoutDst_>
struct TileCopyTlaExt<Catlass::Arch::AtlasA2, 
    tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc_, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst_, AscendC::TPosition::A1>,
    Catlass::layout::ColumnMajor, Catlass::layout::nZ> {
    using LayoutDst = LayoutDst_;
    using LayoutSrc = LayoutSrc_;
    using TensorDst = tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, AscendC::TPosition::A1>;
    using TensorSrc = tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, AscendC::TPosition::GM>;
    using ActualShape = tla::Shape<uint32_t, uint32_t>;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(ElementSrc);

    // Mehtods

    CATLASS_DEVICE
    TileCopyTlaExt() {};

    CATLASS_DEVICE
    void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, ActualShape actualShape)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = tla::get<0>(actualShape);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = tla::get<0, 1>(dstTensor.stride()) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        intriParams.nValue = tla::get<1>(actualShape);
        intriParams.srcDValue = tla::get<1>(srcTensor.stride());
        intriParams.dstNzNStride = tla::get<1, 0>(dstTensor.stride()) / ELE_NUM_PER_C0;
        AscendC::DataCopy(dstTensor.data(), srcTensor.data(), intriParams);
    }
};

/// Partial specialization for TileCopyTlaExt, CopyGmToL1, AtlasA2, PaddingColumnMajor in and nZ out.
template <class ElementSrc, class ElementDst, class LayoutSrc_, class LayoutDst_>
struct TileCopyTlaExt<Catlass::Arch::AtlasA2,
    tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc_, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst_, AscendC::TPosition::A1>,
    Catlass::layout::PaddingColumnMajor, Catlass::layout::nZ> {
    using LayoutDst = LayoutDst_;
    using LayoutSrc = LayoutSrc_;
    using TensorDst = tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, AscendC::TPosition::A1>;
    using TensorSrc = tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, AscendC::TPosition::GM>;
    using ActualShape = tla::Shape<uint32_t, uint32_t>;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(ElementSrc);

    // Mehtods

    CATLASS_DEVICE
    TileCopyTlaExt() {};

    CATLASS_DEVICE
    void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, ActualShape actualShape)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = tla::get<0>(actualShape);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = tla::get<0, 1>(dstTensor.stride()) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        intriParams.nValue = tla::get<1>(actualShape);
        intriParams.srcDValue = tla::get<1, 0>(srcTensor.stride());
        intriParams.dstNzNStride = tla::get<1, 0>(dstTensor.stride()) / ELE_NUM_PER_C0;
        AscendC::DataCopy(dstTensor.data(), srcTensor.data(), intriParams);
    }
};

template <class ArchTag, class Element>
struct CopyGmToL1<ArchTag, Catlass::Gemm::GemmType<Element, Catlass::layout::VectorLayout, AscendC::TPosition::GM>,
    Catlass::Gemm::GemmType<Element, Catlass::layout::VectorLayout, AscendC::TPosition::A1>> {
    using LayoutDst = Catlass::layout::VectorLayout;
    using LayoutSrc = Catlass::layout::VectorLayout;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = layoutDst.shape(0) / ELE_NUM_PER_C0;
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        AscendC::DataCopy(dstTensor, srcTensor, intriParams);
    }
};
/////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace CubeSelf::Gemm::Tile

#endif