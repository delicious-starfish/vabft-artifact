

#ifndef CATLASS_GEMM_TILE_TILE_MMAD_HPP_SELF
#define CATLASS_GEMM_TILE_TILE_MMAD_HPP_SELF

# include "catlass/catlass.hpp"
# include "catlass/gemm/helper.hpp"
# include "tla/tensor.hpp"

namespace CubeSelf::Gemm::Tile{

///////////////////////////////////////////////////////////

template <
    /// Tag indicating architecture
    class ArchTag_,
    /// GemmType for A matrix operand
    class AType_,
    /// GemmType type for B matrix operand
    class BType_,
    /// GemmType type for Bias operand
    class BiasType_
>
struct TileMmad {
    using ElementA = typename AType_::Element;
    using ElementB = typename BType_::Element;
    using ElementAccumulator = typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA,ElementB>::ElementAccumulator;


    // Methods
    CATLASS_DEVICE
    TileMmad() {}

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<ElementAccumulator> const & l0CTensor,
    AscendC::LocalTensor<ElementA> const & l0ATensor,
    AscendC::LocalTensor<ElementB> const & l0BTensor,
    uint32_t m, uint32_t n, uint32_t k, bool initC=true, uint8_t unitFlag = 0)
    {
        AscendC::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;

        mmadParams.unitFlag = unitFlag;
        mmadParams.cmatrixInitVal = initC;

        if constexpr (std::is_same_v<ElementA, float> && std::is_same_v<typename AType_::Layout, Catlass::layout::ColumnMajor>) {
            mmadParams.kDirectionAlign = true;
        }

        AscendC::Mmad(l0CTensor,
                      l0ATensor,
                      l0BTensor,
                      mmadParams);
        
        const uint32_t PIPE_M_BARRIER_THRESHOLD = 10;
        if((m / Catlass::C0_NUM_PER_FRACTAL) * (n / Catlass::C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD)
        {
            AscendC::PipeBarrier<PIPE_M>();
        }
    }

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<ElementAccumulator> & l0CTensor,
        AscendC::LocalTensor<ElementA> & l0ATensor,
        AscendC::LocalTensor<ElementB> & l0BTensor,
        AscendC::LocalTensor<ElementAccumulator> & l0BiasTensor,
        uint32_t m, uint32_t n, uint32_t k, 
        bool initC=true, uint8_t unitFlag = 0)
    {
        AscendC::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        mmadParams.unitFlag = unitFlag;

        mmadParams.cmatrixInitVal = false;

        if constexpr (std::is_same_v<ElementA, float> && std::is_same_v<typename AType_::Layout, Catlass::layout::ColumnMajor>) {
            mmadParams.kDirectionAlign = true;
        }

        AscendC::Mmad(l0CTensor,
                      l0ATensor,
                      l0BTensor,
                      l0BiasTensor,
                      mmadParams);
        
        const uint32_t PIPE_M_BARRIER_THRESHOLD = 10;
        if((m / Catlass::C0_NUM_PER_FRACTAL) * (n / Catlass::C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD)
        {
            AscendC::PipeBarrier<PIPE_M>();
        }
    }
};

///////////////////////////////////////////TileMmadTla/////////////////////////////////////////////////
template <
    /// Tag indicating architecture
    class ArchTag_,
    /// tla::Tensor type for A matrix operand
    class TensorA,
    /// tla::Tensor type for B matrix operand
    class TensorB,
    /// tla::Tensor type for C matrix operand
    class TensorC,
    /// tla::Tensor type for Bias operand
    class TensorBias = void
>
struct TileMmadTla{
    // Method

    CATLASS_DEVICE
    TileMmadTla(){}

    CATLASS_DEVICE
    void operator()(TensorC const & l0CTensor, 
        TensorA const & l0ATensor, 
        TensorB const & l0BTensor,
        uint32_t m, uint32_t n, uint32_t k,
        bool initC=true, uint8_t unitFlag=0)
    {
        AscendC::MmadParams mmadParams;

        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;

        mmadParams.unitFlag = unitFlag;
        mmadParams.cmatrixInitVal = initC;

        AscendC::Mmad(l0CTensor.data(),
                      l0ATensor.data(),
                      l0BTensor.data(),
                      mmadParams);
        
        const uint32_t PIPE_M_BARRIER_THRESHOLD = 10;

        if((m / Catlass::C0_NUM_PER_FRACTAL) * (n / Catlass::C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD)
        {
            AscendC::PipeBarrier<PIPE_M>();
        }
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // CubeSelf::Gemm::Tile
#endif