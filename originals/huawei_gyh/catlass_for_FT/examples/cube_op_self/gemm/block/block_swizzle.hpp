#ifndef CATLASS_GEMM_BLOCK_BLOCK_SWIZZLE_HPP_SELF
#define CATLASS_GEMM_BLOCK_BLOCK_SWIZZLE_HPP_SELF

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
            // 奇数编号的tileBlock, 在 column 上从右向左进行swizzle 块的滑动
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
            /*
            loopsMNK.k(): 即在K维度上一共有多少个L1Tile
            splitkFactor: 即在K维度上一共分成多少份 Slice
            而splitKSliceIdx则指的是当前为第几个K Slice
            若splitkFactor 小于loopsMNK.k() % splitkFactor这个余数，
            则代表了当前的Slice要比之后的Slice多一个Tile，因为前余数个Slice多补了1个Tile这样一来就可以分摊掉多余的K，
            可以进一步均衡负载，且避免掉不整除的余数问题导致的性能波动
            与此同时在开头的slice中补全时，每个补的L1Tile的K都是满的，
            这样一来性能可控且负载更均衡，因为只有最后一个
            Slice上可能存在一个不满的L1Tile。
            因此这样一来，其在K维度上处理的L1Tile的数量即为：
            1) loopsMNK.k() / splitkFactor：向下取整后，即每个Slice 均需要平均分摊的L1Tile K的下限
            2）1：因为有余数，所以提前分摊，多处理一个L1 Tile K (因为余数或者说无法均分的L1Tile的量一定不超过Slice K的数量，多分摊一个即可)
            因此，对于当前 Slice 而言，其在K上的偏移为（因为前面的Slice也要处理相同数量的L1 Tile）:
            每个前面Slice的L1 Tile的量：（loopsMNK.k() / splitkFactor + 1） x splitkSliceIdx
            */
            return (loopsMNK.k() / splitkFactor + 1) * splitkSliceIdx;
        } else {
            /*
            若K Slice的Id，splitKSliceIdx 大于余数loopsMNK.k() % splitkFactor，
            则代表着是后面的Slice，即处理后面部分L1Tile K的K slice。
            在这种情况下，前面的Slice 已经将余数的L1Tile 完全处理完了
            因此在这种情况下，后面的每个Slice只需要处理loopsMNK.k() / splitkFactor 个 L1Tile即可了
            而前面的Slice已经将无法整除的，多余的Tile都分摊完了
            所以后面只需要每次处理loopsMNK.k() / splitkFactor个Tile了
            那么其偏移即为：
            前面全部的Slice处理的基础loopsMNK.k() / splitkFactor 个Tile x splitkSliceIdx
             + 
            所有无法整除的余数L1 Tile：loopsMNK.k() % splitkFactor
            计算后则为：
            */
            return splitkSliceIdx * (loopsMNK.k() / splitkFactor) + loopsMNK.k() % splitkFactor;
        }
    }

    /*
    该函数用来判断当前处于第几个K Slice：
    此处调度的逻辑中，是先处理完一个K slice全部的 M,N上的Tile/Block，然后
    再处理下一个K slice，即K上的下一部分，因此K是最外层的调度侧。
    */
    CATLASS_DEVICE
    uint32_t GetSplitkSliceIdx(uint32_t taskIdx) const
    {
        /*
        首先，计算在M,N上面需要进行多少Loop/Block，与No-split K的情况一致，只是K上的Tile数量减少了
        */
        uint32_t mnLoops = loopsMNK.m() * loopsMNK.n();
        /*
        然后，我们求出来总的迭代次数，即GetCoreLoops()
        在此基础上，可以获取当前的核处理的taske Id对应了第几个Block/Loop，
        此处就是直接取余，保证不少于全部Block的规模即可
        最后，直接除以Slice上所需的MN Block的数量，即代表此时进行到的Slice即可。
        */
        return taskIdx % GetCoreLoops() / mnLoops;
    }

    /*
        计算在分K进行Slice后整个矩阵乘所需的迭代或者说Block数量
        即：每个K Slice 在M,N上的Block数量 x K Slice的数量
    */

    CATLASS_DEVICE
    uint32_t GetCoreLoops() const
    {
        return loopsMNK.m() * loopsMNK.n() * splitkFactor; // 在 K 轴上一共只分为 splitkFactor 个，每次在迭代中处理完所有相应的K轴元素
    }


    /*
    获取 Batch的数量，即做几个AXB矩阵的乘法，因为某些情况下，
    是可以于一次批量完成多组矩阵乘法的，假设一组包含n个A和B的pair，
    则每次A_i x B_i需要GetCoreLoops() 次，因此需要判断当前是计算到哪一组矩阵乘了
    因此直接让当前的taskIdx / GetCoreLoops() 即可。
    */
    CATLASS_DEVICE
    uint32_t GetBatchIdx(uint32_t taskIdx)
    {
        return taskIdx / GetCoreLoops();
    }

    /*
    计算当前具体Block对应A,B矩阵在M,N,K维度上的具体坐标，单位为L1Tile
    这是矩阵乘法调度器的核心函数和基础
    */
    CATLASS_DEVICE
    Catlass::GemmCoord GetBlockCoord(uint32_t taskIdx){
        /*
        先根据每个K Slice 内部的MN Block 数量，计算出来当前的迭代ID属于第几个K slice
        */
        uint32_t splitkSliceIdx = GetSplitkSliceIdx(taskIdx);
        /*
        根据所属于的 K Slice的ID，直接计算当前Slice在K维度上对应的起始坐标，即若为前面的Slice，需要分摊无法被均分的，即
        向下取整后多出来的 L1Tile K 则在前面的 Slice 中进行分摊，若为后面的Slice，则仅计算向下取整的L1 Tile K部分即可。
        */
        uint32_t kIdx = GetKIdxBySplitkSliceIdx(splitkSliceIdx);
        /*
        在调度时，K的Slice位于调度最外层，每个Slice内部，则在M,N上的Block内计算一定数量的L1 Tile。
        因此在确定了最外层的K的坐标后，则需要计算Slice内部在M,N维度上的Block的坐标，以L1 Tile为单位
        因此，我们先计算了内部的总线性坐标，取余每个Slice内的Block数量(loopsMNK.m() * loopsMNK.n())即可：
        */
        uint32_t innerIdx = taskIdx % (loopsMNK.m() * loopsMNK.n());

        if constexpr (SwizzleDirection == 0) {
            // Zn:即在每个Block的Swizzle 区域内部，先沿着行（即M维度，列主序）进行调度，
            // 然后在每个Swizzle区域之间，沿着列进行（即N维度，行主序）
            // 但是在这里，我们每次将一整行都作为一个Swizzle Block内部，所以等效于Swizzle内部全部的列之间都是列主序，
            // 然后多个行间进行了Swizzle Block的划分
            // 所以实质上这里等效于nN了（即整个行均在一个Swizzle Block里面）
            // 每个Slice内部有多少个Swizzle Block区域，即按行的Tile 数进行分区（每个Block内处理固定量的行内的全部Tile）
            uint32_t tileBlockLoop = CeilDiv(loopsMNK.m(), SwizzleOffset);
            // 判断当前的task ID的迭代对应的Tile Block 处于哪个Swizzle Block中，
            // 每个Swizzle Block对应SwizzleOffset * N维度上的L1 Tile 数量的Tile
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMNK.n());
            // 判断在每个Swizzle Block内部，当前Task ID对应的迭代对应了Swizzle Block内的第几个具体的 L1 Tile Block
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMNK.n());
            // 当前Swizzle Block内，处理的行数为SwizzleOffset：
            uint32_t nRow = SwizzleOffset;
            // 若当前处于最后一个Swizzle Block的区域内部，显然对应的行数可能不到SwizzleOffset，调整为剩余的行数即可：
            if (tileBlockIdx == tileBlockLoop - 1) {
                nRow = loopsMNK.m() - SwizzleOffset * tileBlockIdx;
            }
            // 确定当前迭代的Task ID所对应的Tile Block的 L1 Tile在M维度上的起始坐标，即当前处于第几个Swizzle Block执行的第几行：
            //前面每次执行时包括了SwizzleOffset个M维度的行，因此可以求出Swizzle Block整体的M维度行上的偏移 
            // +
            // 在Swizzle Block内部，按照n，即列主序，优先处理一列上的全部行 Tile Block，所以每次取余行数，确定当前进行到某一列的第几行了
            uint32_t mIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nRow;
            // 判断在每个Swizzle Block内部，按照列主序的方式，当前进行到第几列， 即处理了多少个nRow的Tile Block了：
            uint32_t nIdx = inTileBlockIdx / nRow;
            // 这里打乱执行顺序，即若为奇数Swizzle Block，则从后面的列上的Tile Block开始执行，避免出现数据在shared memory中的冲突？
            if (tileBlockIdx % 2 == 1) {
                nIdx = loopsMNK.n() - nIdx - 1;
            }
            // 返回了当前Task ID对应的Tile Block的其实坐标即可：
            return Catlass::GemmCoord{mIdx, nIdx, kIdx};

        } else if constexpr (SwizzleDirection == 1){
            // Nz / zN
            // Nz/ zN: 即在每个Block的Swizzle 区域内部，先沿着列（即N维度，行主序）进行调度，
            // 然后在每个Swizzle区域之间，沿着行进行（即M维度，列主序）
            // 但是在这里，我们每次将一整列都作为一个Swizzle Block内部，所以等效于Swizzle内部全部的行之间都是行主序，
            // 然后多个列间进行了Swizzle Block的划分
            // 所以实质上这里等效于zZ了（即整个列均在一个Swizzle Block里面）

            // 每个Slice内部有多少个Swizzle Block区域，即按列的Tile 数进行分区（每个Block内处理固定量的列内的全部Tile）
            uint32_t tileBlockLoop = CeilDiv(loopsMNK.n(), SwizzleOffset);
            // 判断当前的task ID的迭代对应的Tile Block 处于哪个Swizzle Block中，
            // 每个Swizzle Block对应SwizzleOffset * M 维度上的L1 Tile 数量的Tile
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMNK.m());
            // 判断在每个Swizzle Block内部，当前Task ID对应的迭代对应了Swizzle Block内的第几个具体的 L1 Tile Block
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMNK.m());

            // 当前Swizzle Block内，处理的列数为SwizzleOffset：
            uint32_t nCol = SwizzleOffset;
            // 若当前处于最后一个Swizzle Block的区域内部，显然对应的列数可能不到SwizzleOffset，调整为剩余的列数即可：
            if (tileBlockIdx == tileBlockLoop - 1) {
                nCol = loopsMNK.n() - SwizzleOffset * tileBlockIdx;
            }

            // 行坐标（M维度）的确定：判断在每个Swizzle Block内部，按照行主序的方式，当前进行到M维度的第几行了， 即处理了多少个nCol的Tile Block了：
            uint32_t mIdx = inTileBlockIdx / nCol;
            // 列坐标（N维度）的确定：
            // 确定当前迭代的Task ID所对应的Tile Block的 L1 Tile在N维度上的起始坐标，即当前处于第几个Swizzle Block执行的第几列：
            //前面每次执行时包括了SwizzleOffset个N维度的列，因此可以求出Swizzle Block整体的N维度列上的偏移 
            // +
            // 在Swizzle Block内部，按照z，即行主序，优先处理一行上的全部列的 Tile Block，所以每次取余列数，
            // 确定当前进行到某一行的第几列了
            uint32_t nIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nCol;
            // 这里打乱执行顺序，即若为奇数Swizzle Block，则从后面的行上的Tile Block开始执行，避免出现数据在shared memory中的冲突？
            if (tileBlockIdx % 2 == 1) {
                mIdx = loopsMNK.m() - mIdx - 1;
            }
            
            return Catlass::GemmCoord{mIdx, nIdx, kIdx};
        }
    }

    /*
    最后，在确定了当前Tile Block的起始坐标（单位为L1 Tile Block）后，
    我们可以确定每个Block中实际的M,N,K上的规模即可，单位维元素
    GetActualBlockShape(GemmCoord blockCoord, uint32_t splitkSliceIdx)
    */
    CATLASS_DEVICE
    Catlass::GemmCoord GetActualBlockShape(Catlass::GemmCoord blockCoord, uint32_t splitkSliceIdx)
    {
        // 首先定义每个Slice 在K上的长度（单位为元素）
        uint32_t splitkSliceLen;

        // 输入K Slice 的ID，对于前面的Slice，即ID < K上的Tile相对于splitKFactor的余数时
        if (splitkSliceIdx < loopsMNK.k() % splitkFactor) {
            // 分摊无法被均分到每个Slice上的剩余的Tile，每个分摊一个即可，因为剩余的数量一定小于Slice的数量
            // 根据Tile的数量，计算出具体的SplitKSliceLen，即Slice在K上的规模：
            // 可均分部分的均分Tile + 分摊 Tile
            splitkSliceLen = (loopsMNK.k() / splitkFactor + 1) * tileShape.k();
        } else {
            // 对于靠后的Slice，即ID >= K上的Tile相对于splitKFactor的余数时
            // 只需要执行可均分部分的均分Tile即可
            splitkSliceLen = (loopsMNK.k() / splitkFactor) * tileShape.k();
        }
        
        // 在M维度上，坐标非最后一个Tile的话就是全部L1Tile大小的M，反之则为最终剩余的M
        uint32_t mActual = (blockCoord.m() == (loopsMNK.m() - 1)) ?
            (problemShape.m() - blockCoord.m() * tileShape.m()) : tileShape.m();

        // 在N维度上，坐标非最后一个Tile的话就是全部L1Tile大小的N，反之则为最终剩余的N
        uint32_t nActual = (blockCoord.n() == (loopsMNK.n() - 1)) ?
            (problemShape.n() - blockCoord.n() * tileShape.n()) : tileShape.n();

        // 在K维度上，坐标非最后一个Slice的话，就是之前预估的理论上的L1TileShape::K的整数倍的length，
        // 反之则为最终剩余的K
        uint32_t kActual = (splitkSliceIdx == (splitkFactor - 1)) ?
            (problemShape.k() - blockCoord.k() * tileShape.k()) : splitkSliceLen;

        return Catlass::GemmCoord{mActual, nActual, kActual};
    }
};

} // namespace CubeSelf::Gemm::Block
#endif