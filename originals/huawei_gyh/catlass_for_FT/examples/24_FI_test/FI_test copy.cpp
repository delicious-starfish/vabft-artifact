#include <iostream>
#include <vector>

#include "helper.hpp"
#include "golden.hpp"

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "catlass/gemv/block/block_gemv.hpp"
#include "catlass/gemv/block/block_sum_aiv_ceft_no_splitk_thre_robust_FI.hpp"

#include "catlass/gemv/device/device_gemv.hpp"

#include "fp16_t.h"
#include "bfloat16.h"



using namespace Catlass;

using fp16_t        = op::fp16_t;
using op_bf16_t     = op::bfloat16;

using GemmInTypeC   = half;
using GemmInTypeN   = half;

using GemmOutTypeC  = float;
using GemmOutTypeN  = float;

using GemvInTypeCforCE      = half;
using GemvInTypeNforCE      = half;
using GemvInTypeCforAB      = half;
using GemvInTypeNforAB      = half;

using GemvOutTypeC  = float;
using GemvOutTypeN  = float;

using ScalarTypeC   = float;
using ScalarTypeN   = float;

using ScalarType    = float;

struct Options{
    const std::string HELPER = "24_FI_test m n k rt beta thre_type e_max red_cores inject_row inject_col inject_bit [device_id]";
    GemmCoord problemGemmShape{128, 128, 128};
    GemvCoord problemShape{128, 128};

    int32_t deviceId{1};

    float round_exp{0.0f};
    float beta{1.0f};

    int thre_type{0};

    uint32_t reduce_cores{8};
    uint32_t inject_row{0};
    uint32_t inject_col{0};
    uint32_t inject_bit{0};

    float e_max;

    Options() = default;

    int Parse(int argc, const char** argv) {
        enum ArgsIndex {
            M_INDEX = 1,
            N_INDEX,
            K_INDEX,
            RT_INDEX,
            BETA_INDEX,
            THRE_TYPE_INDEX,
            E_MAX_INDEX,
            RED_CORES_INDEX,
            INJECT_ROW_INDEX,
            INJECT_COL_INDEX,
            INJECT_BIT_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };
        if (argc > ARGS_MAX || argc < N_INDEX) {
            std::cerr << HELPER << std::endl;
            return -1;
        }
        problemGemmShape.m()    = std::atoi(argv[M_INDEX]);
        problemGemmShape.n()    = std::atoi(argv[N_INDEX]);
        problemGemmShape.k()    = std::atoi(argv[K_INDEX]);

        problemShape.m()        = std::atoi(argv[M_INDEX]);
        problemShape.n()        = std::atoi(argv[N_INDEX]);

        round_exp               = static_cast<float>(std::stof(argv[RT_INDEX]));
        beta                    = static_cast<float>(std::stof(argv[BETA_INDEX]));

        thre_type               = std::atoi(argv[THRE_TYPE_INDEX]);

        e_max                   = static_cast<float>(std::stof(argv[E_MAX_INDEX]));

        inject_row              = std::atoi(argv[INJECT_ROW_INDEX]);
        inject_col              = std::atoi(argv[INJECT_ROW_INDEX]);
        inject_bit              = std::atoi(argv[INJECT_BIT_INDEX]);

        if (argc == ARGS_MAX) {
            deviceId = std::atoi(argv[DEVICE_ID_INDEX]);
        }
        return 0;
    }
};

void Run(Options options){
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    using L1TileShape = GemmShape<128, 256, 256>;
    using L0TileShape = GemmShape<128, 256, 64>;

    static constexpr uint32_t BLK_BYTE      = 32; // Ascend通用框架下的内存分块大小
    static constexpr uint32_t BLK_CELEC     = BLK_BYTE / sizeof(GemvOutTypeC); // 通用内存块下能够放下几个C (C Element Count)
    uint32_t    m   = options.problemShape.m();
    uint32_t    n   = options.problemShape.n();
    uint32_t    k   = options.problemGemmShape.k();

    uint32_t    splitNnum   = (n + L1TileShape::n - 1) / L1TileShape::n;
    
    GemvCoord   problemShapeCol{m, n};

    GemvCoord   problemShapeBe{k, n};   // B: <k, n>, e     <n, 1>
    GemvCoord   problemShapeABe{m, k};  // A: <m, k>, Be    <k, 1>

    GemvCoord   problemShapeETA{k, m};  // 看不懂，为什么是倒过来的
    GemvCoord   problemShapeETAB{n, k}; // 真神奇，这个也是倒过来的

    size_t      lenA    = static_cast<size_t>(m) * k;
    size_t      lenB    = static_cast<size_t>(k) * n;
    size_t      lenC    = static_cast<size_t>(m) * n;

    // 容错计算相关
    size_t      lenARed         = static_cast<size_t>(m) * 1; // 每行一个
    size_t      lenBMean_P      = (splitNnum + BLK_CELEC - 1) / BLK_CELEC; // B全局统计量的缓存区长度
    size_t      lenBMax_P       = lenBMean;
    size_t      lenBMin_P       = lenBMean;
    size_t      lenBMeanAbs_P   = lenBMean;
    size_t      lenBMeanSquare_P= lenBMean;
    size_t      lenBVar_P       = lenBMean;
    size_t      lenThre = static_cast<size_t>(m);       // 正确实现中就它一个，说明最后容错计算没有拆开

    size_t      lenCOMPCol = (static_cast<size_t>(n) + 8 - 1) / 8; // 似乎代表着错误检出判断是以8个为一组的?
    size_t      lenCOMPRow = (static_cast<size_t>(m) + 8 - 1) / 8; // 似乎代表着错误检出判断是以8个为一组的?

    // 计算用的常量向量
    size_t      lenX        = static_cast<size_t>(n) * 1; // 用于计算CE, CE = C \cdot X
    size_t      lenXV       = static_cast<size_t>(n) * 1; // 用于计算BE, BE = B \cdot XV
    size_t      lenXV_AMean = static_cast<size_t>(k) * 1; // 用于计算A的均值
    
    // 错误检出中的结果向量以及中间向量
    // 因为每一个NSlice都能够得到一个列和。于是需要额外乘上splitNnum
    // P -> process 表示计算时占用的空间
    size_t      lenABE_P      = static_cast<size_t>(m) * splitNnum; // 用于存储ABE的结果. 即lenD
    size_t      lenCE_P       = static_cast<size_t>(m) * splitNnum; // 用于储存CE的结果. 即lenZ
    size_t      lenBE_P       = static_cast<size_t>(k) * splitNnum;



    // c_代表在AIC上占据空间, v_代表在AIV上占据空间
    size_t      c_sizeA     = lenA * sizeof(GemmInTypeC);
    size_t      c_sizeB     = lenB * sizeof(GemmInTypeC);
    size_t      c_sizeC     = lenC * sizeof(GemmOutTypeC);
    size_t      v_sizeBE_P  = lenBE_P * sizeof(GemvInTypeCforAB);

    size_t      v_sizeCE_P   = lenCE_P * sizeof(GemvOutTypeC);
    size_t      v_sizeBE_P   = lenBE_P * sizeof(GemvOutTypeC);
    size_t      v_sizeABE_P  = lenABE_P * sizeof(GemvOutTypeC);

    size_t      v_sizeBMeanAbs_P    = lenBMeanAbs_P * sizeof(GemvOutTypeC);
    size_t      v_sizeBMeanSquare_P = lenBMeanSquare_P * sizeof(GemvOutTypeC);
    size_t      v_sizeAMean         = lenARed * sizeof(GemvOutTypeC);
    size_t      v_sizeAMin          = lenARed * sizeof(GemvOutTypeC);
    size_t      v_sizeAMax          = lenARed * sizeof(GemvOutTypeC);

    size_t      v_sizeThre  = lenThre * sizeof(GemvOutTypeC);   // 需要和Ce一致, 用别的精度会导致损失？

    size_t      sizeCOMPCol = lenCOMPCol * sizeof(uint8_t); // 不清楚在哪个设备上
    size_t      sizeCOMPRow = lenCOMPRow * sizeof(uint8_t); // 不清楚在哪个设备上

    using LayoutCOMP    = layout::VectorLayout;

    using LayoutA       = layout::RowMajor;
    using LayoutAC      = layout::ColMajor;// A Column
    using LayoutB       = layout::RowMajor;
    using LayoutBC      = layout::ColMajor;// B Column
    using LayoutC       = layout::RowMajor;
    using LayoutCC      = layout::ColMajor;// C Column

    LayoutA     layoutA{m, k};
    LayoutB     layoutB{k, n};
    LayoutC     layoutC{m, n};

    LayoutAC    layoutAC{k, m};
    LayoutBC    layoutBC{n, k};
    LayoutCC    layoutCC{n, m};

    ScalarType  alpha{1.0};
    ScalarType  beta{0.0};

    float sum_base  = 1.0;
    float mean_base = 1.0f / (1.0f * k);

    std::vector<uint8_t> hostCOMPRow(lenCOMPRow, 0);
    std::vector<uint8_t> hostCOMPCol(lenCOMPCol, 0);

    std::vector<GemmInTypeC> hostA(lenA);
    std::vector<GemmInTypeC> hostB(lenB);
    std::vector<GemmInTypeC> hostC(lenC, 0.0);


    golden::FillRandomData(hostA, -1.0f, 1.0f);
    gloden::FillRandomData(hostB, -1.0f, 1.0f);

    uint8_t* deviceC{nullptr};
    uint8_t* deviceB{nullptr};
    uint8_t* deviceA{nullptr};

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceA), c_sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceB), c_sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceC), c_sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    ACL_CHECK(aclrtMemcpy(deviceA, c_sizeA, hostA.data(), c_sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceB, c_sizeB, hostA.data(), c_sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceC, c_sizeC, hostA.data(), c_sizeA, ACL_MEMCPY_HOST_TO_DEVICE));


}

int main(int argc, const char** argv) {

    return 0;
}