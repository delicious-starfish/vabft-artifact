#ifndef CATLASS_GEMM_BLOCK_BLOCK_SWIZZLE_HPP_FT_NOSPLITK_SELF
#define CATLASS_GEMM_BLOCK_BLOCK_SWIZZLE_HPP_FT_NOSPLITK_SELF

#include "catlass/catlass.hpp"
#include "catlass/detail/alignment.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace CubeSelf::Gemm::Block{

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Block swizzling function for Gemms
template <uint32_t SwizzleOffset = 1, uint32_t SwizzleDirection = 0>
struct GemmIdentityBlockSwizzle {
    /// Data members

    Catlass::GemmCoord problemShape;
    Catlass::MatrixCoord tileMN;
    Catlass::MatrixCoord loopsMN;

    /// Methods

    CATLASS_DEVICE
    GemmIdentityBlockSwizzle() {}

    CATLASS_DEVICE
    GemmIdentityBlockSwizzle(Catlass::GemmCoord const &problemShape_, Catlass::MatrixCoord const &tileMN_) 
        : problemShape(problemShape_), tileMN(tileMN_)
    {
        loopsMN = CeilDiv(Catlass::MatrixCoord(problemShape.GetCoordMN()), tileMN);   
    }

    CATLASS_DEVICE
    GemmIdentityBlockSwizzle(Catlass::GemmCoord const &problemShape_,
        Catlass::MatrixCoord const &tileMN_, Catlass::MatrixCoord const &loopsMN_) 
        : problemShape(problemShape_), tileMN(tileMN_), loopsMN(loopsMN_) {}

    CATLASS_DEVICE
    void Update(Catlass::GemmCoord const &problemShape_, 
        Catlass::MatrixCoord const &tileMN_)
    {
        problemShape = problemShape_;
        tileMN = tileMN_;

        loopsMN = CeilDiv(Catlass::MatrixCoord(problemShape.GetCoordMN()), tileMN);
    }
    
    CATLASS_DEVICE
    void Update(Catlass::GemmCoord const &problemShape_, 
        Catlass::MatrixCoord const &tileMN_, Catlass::MatrixCoord const &loopsMN_)
    {
        problemShape = problemShape_;
        tileMN = tileMN_;
        loopsMN = loopsMN_;
    }

    CATLASS_DEVICE
    uint32_t GetCoreLoops() const
    {
        return loopsMN.row() * loopsMN.column();
    }

    CATLASS_DEVICE
    uint32_t GetBatchIdx(uint32_t taskIdx)
    {
        return taskIdx / (GetCoreLoops());
    }

    CATLASS_DEVICE
    Catlass::GemmCoord GetBlockCoord(uint32_t taskIdx)
    {
        uint32_t innerIdx = taskIdx % GetCoreLoops();

        if constexpr (SwizzleDirection == 0) {
            // Zn / nZ，即进行swizzle时 TileBlock间外部为 Z 内部为 n：
            uint32_t tileBlockLoop = CeilDiv(loopsMN.row(),SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMN.column());
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMN.column());

            uint32_t nRow = SwizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                // 最后一个Tile block的数据，此时该block对应的行数可能不是swizzleOffset
                nRow = loopsMN.row() - SwizzleOffset * tileBlockIdx;
            }
            // 确定当前loop中起始的行编号，包括：
            /*
            1. 做了多少个tileblock，每个tileblock包括SwizzleOffset个行
            2. 在当前的tileblock中，目前已经做了多少行了，在每个tileblock内部，
               每次做的顺序是 Swizzle_Offset x Swizzle_Offset 个L1 tile或者说循环内部
               做运算，然后进行下一个 Swizzle 块，因此其内部ID是取余 nRows的： 
            */
            uint32_t mIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nRow;
            uint32_t nIdx = inTileBlockIdx / nRow; // 每次swizzle 块中，先 M 后 N。

            // 偶数编号的tileBlock，在 column 上从左向右进行swizzle 块的滑动
            // 奇数编号的tileBlock, 在 
            if(tileBlockIdx % 2 == 1){
                nIdx = loopsMN.column() - nIdx - 1;
            }

            return Catlass::GemmCoord({mIdx,nIdx,0});
        } else if constexpr (SwizzleDirection == 1){
            // Nz/zN: 即swizzle时，TileBlock 间外部为N，内部为z:
            uint32_t tileBlockLoop = CeilDiv(loopsMN.column(),SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMN.row());
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMN.row());

            uint32_t nCol = SwizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nCol = loopsMN.column() - SwizzleOffset * tileBlockIdx;
            }

            uint32_t mIdx = inTileBlockIdx / nCol;
            uint32_t nIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nCol;

            if (tileBlockIdx % 2 == 1) {
                mIdx = loopsMN.row() - mIdx - 1;
            }

            return Catlass::GemmCoord({mIdx, nIdx, 0});
        }
    }

    CATLASS_DEVICE
    Catlass::GemmCoord GetActualBlockShape(Catlass::GemmCoord blockCoord)
    {
        uint32_t mActual = (blockCoord.m() == (loopsMN.row() - 1)) ?
            (problemShape.m() - blockCoord.m() * tileMN.row()) : tileMN.row();
        
        uint32_t nActual = (blockCoord.n() == (loopsMN.column() - 1)) ?
            (problemShape.n() - blockCoord.n() * tileMN.column()) : tileMN.column();
        
        uint32_t kActual = problemShape.k();

        return Catlass::GemmCoord{mActual, nActual, kActual};
    }
};

/// Block swizzling function for Splitk Gemms
template <uint32_t SwizzleOffset = 1, uint32_t SwizzleDirection = 0>
struct SplitkGemmIdentityBlockSwizzle{
    //// Data Members

    Catlass::GemmCoord problemShape;
    Catlass::GemmCoord tileShape;
    Catlass::GemmCoord loopsMNK; // 需要split-k

    uint32_t splitkFactor = 1;  // splite k dim into virtual cores

    /// Methods

    CATLASS_DEVICE
    SplitkGemmIdentityBlockSwizzle() {}

    CATLASS_DEVICE
    SplitkGemmIdentityBlockSwizzle(
        Catlass::GemmCoord const &problemShape_, Catlass::GemmCoord const &tileShape_,
        uint32_t splitkFactor_ = 1) : problemShape(problemShape_), tileShape(tileShape_), splitkFactor(splitkFactor_)
    {
       loopsMNK = CeilDiv(problemShape, tileShape);
    }

    CATLASS_DEVICE
    uint32_t GetKIdxBySplitkSliceIdx(uint32_t splitkSliceIdx) const
    {
        if (splitkSliceIdx < (loopsMNK.k() % splitkFactor)) {
            return (loopsMNK.k() / splitkFactor + 1) * splitkSliceIdx;
        } else {
            return splitkSliceIdx * (loopsMNK.k() / splitkFactor) + loopsMNK.k() % splitkFactor;
        }
    }

    CATLASS_DEVICE
    uint32_t GetSplitkSliceIdx(uint32_t taskIdx) const
    {
        uint32_t mnLoops = loopsMNK.m() * loopsMNK.n();

        return taskIdx % GetCoreLoops() / mnLoops;
    }

    CATLASS_DEVICE
    uint32_t GetCoreLoops(){
        return loopsMNK.m() * loopsMNK.n() * splitkFactor; // 在 K 轴上一共只分为 splitkFactor 个，每次在迭代中处理完所有相应的K轴元素
    }

    CATLASS_DEVICE
    uint32_t GetBatchIdx(uint32_t taskIdx)
    {
        return taskIdx / GetCoreLoops();
    }

    CATLASS_DEVICE
    Catlass::GemmCoord GetBlockCoord(uint32_t taskIdx){
        uint32_t splitkSliceIdx = GetSplitkSliceIdx(taskIdx);
        uint32_t kIdx = GetKIdxBySplitkSliceIdx(splitkSliceIdx);

        uint32_t innerIdx = taskIdx % (loopsMNK.m() * loopsMNK.n());

        if constexpr (SwizzleDirection == 0) {
            // Zn / nZ:
            uint32_t tileBlockLoop = CeilDiv(loopsMNK.m(), SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMNK.n());
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMNK.n());

            uint32_t nRow = SwizzleOffset;

            if (tileBlockIdx == tileBlockLoop - 1) {
                nRow = loopsMNK.m() - SwizzleOffset * tileBlockIdx;
            }

            uint32_t mIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nRow;
            uint32_t nIdx = inTileBlockIdx / nRow;

            if (tileBlockIdx % 2 == 1) 
            {
                nIdx = loopsMNK.n() - nIdx - 1;
            }

            return Catlass::GemmCoord{mIdx, nIdx, kIdx};

        } else if constexpr (SwizzleDirection == 1){
            // Nz / zN
            uint32_t tileBlockLoop = CeilDiv(loopsMNK.n(), SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMNK.m());
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMNK.m());

            uint32_t nCol = SwizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nCol = loopsMNK.n() - SwizzleOffset * tileBlockIdx;
            }

            uint32_t mIdx = inTileBlockIdx / nCol;
            uint32_t nIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nCol;

            if (tileBlockIdx % 2 == 1) {
                mIdx = loopsMNK.m() - mIdx - 1;
            }
            
            return Catlass::GemmCoord{mIdx, nIdx, kIdx};
        }
    }

    CATLASS_DEVICE
    Catlass::GemmCoord GetActualBlockShape(Catlass::GemmCoord blockCoord, uint32_t splitkSliceIdx)
    {
        uint32_t splitkSliceLen;

        if (splitkSliceIdx < loopsMNK.k() % splitkFactor) {
            splitkSliceLen = (loopsMNK.k() / splitkFactor + 1) * tileShape.k();
        } else {
            splitkSliceLen = (loopsMNK.k() / splitkFactor) * tileShape.k();
        }

        uint32_t mActual = (blockCoord.m() == (loopsMNK.m() - 1)) ?
            (problemShape.m() - blockCoord.m() * tileShape.m()) : tileShape.m();
        
        uint32_t nActual = (blockCoord.n() == (loopsMNK.n() - 1)) ?
            (problemShape.n() - blockCoord.n() * tileShape.n()) : tileShape.n();

        uint32_t kActual = (splitkSliceIdx == (splitkFactor - 1)) ?
            (problemShape.k() - blockCoord.k() * tileShape.k()) : splitkSliceLen;

        return Catlass::GemmCoord{mActual, nActual, kActual};
    }
};

} // namespace CubeSelf::Gemm::Block
#endif