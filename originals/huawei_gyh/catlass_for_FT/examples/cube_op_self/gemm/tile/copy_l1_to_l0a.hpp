
#ifndef CATLASS_GEMM_TILE_COPY_L1_TO_L0A_HPP_SELF
#define CATLASS_GEMM_TILE_COPY_L1_TO_L0A_HPP_SELF

// catlass/

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "gemm/tile/tile_copy_tla.hpp"
#include "tla/tensor.hpp"

namespace CubeSelf::Gemm::Tile{

template<
    class ArchTag,
    class L1Type,
    class L0Type=void
>
struct CopyL1ToL0A {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to l0, can not find the specialization.");
};

////////////////////////////////
/// new add gemm
template<class ArchTag,class Element>
struct CopyL1ToL0A<ArchTag,
    Catlass::Gemm::GemmType<Element, Catlass::layout::zN, AscendC::TPosition::A1>,
    Catlass::Gemm::GemmType<Element, Catlass::layout::zZ, AscendC::TPosition::A2>>
{
    using LayoutDst = Catlass::layout::zZ;
    using LayoutSrc = Catlass::layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 =  Catlass::BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = Catlass::BYTE_PER_FRACTAL / sizeof(Element);

    CATLASS_DEVICE
    CopyL1ToL0A(){}

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::LocalTensor<Element> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
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
struct CopyL1ToL0A<ArchTag,
    Catlass::Gemm::GemmType<Element,Catlass::layout::nN, AscendC::TPosition::A1>,
    Catlass::Gemm::GemmType<Element,Catlass::layout::zZ, AscendC::TPosition::A2>>
{
    using LayoutDst = Catlass::layout::zZ;
    using LayoutSrc = Catlass::layout::nN;

    static constexpr uint32_t ELE_NUM_PER_C0 =  Catlass::BYTE_PER_C0 / sizeof(Element);

    CATLASS_DEVICE
    CopyL1ToL0A(){}

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::LocalTensor<Element> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<Catlass::C0_NUM_PER_FRACTAL>(layoutDst.orgShape(1)));
        loadDataParams.srcStride = static_cast<uint16_t>(CeilDiv<ELE_NUM_PER_C0>(layoutSrc.orgShape(0)));;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = true;
        loadDataParams.addrMode = 0;
        for(uint32_t i = 0; i < CeilDiv<ELE_NUM_PER_C0>(layoutSrc.orgShape(0)); i++){
            AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], 
                srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};

template<class ArchTag>
struct CopyL1ToL0A<ArchTag,
    Catlass::Gemm::GemmType<float, Catlass::layout::nN, AscendC::TPosition::A1>,
    Catlass::Gemm::GemmType<float, Catlass::layout::zZ, AscendC::TPosition::A2>
>{
    using Element = float;
    using LayoutDst = Catlass::layout::zZ;
    using LayoutSrc = Catlass::layout::nN;

    static constexpr uint32_t ELE_NUM_PER_C0 =  Catlass::BYTE_PER_C0 / sizeof(Element);

    CATLASS_DEVICE
    CopyL1ToL0A(){}

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::LocalTensor<Element> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        AscendC::LoadData2dTransposeParams loadDataParams;
        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<Catlass::C0_NUM_PER_FRACTAL>(layoutDst.orgShape(1)));
        loadDataParams.srcStride = static_cast<uint16_t>(CeilDiv<Catlass::C0_NUM_PER_FRACTAL>(layoutSrc.orgShape(0)));
        loadDataParams.dstGap = 1;
        loadDataParams.dstFracGap = 0;
        for(uint32_t i = 0; i < CeilDiv<Catlass::C0_NUM_PER_FRACTAL>(layoutSrc.orgShape(0)); i++){
            AscendC::LoadDataWithTranspose(dstTensor[i * layoutDst.stride(1)],
                srcTensor[i * layoutSrc.stride(1) * 2], loadDataParams);
        }

        /*
        I. 函数原型
        template <typename T>
        __aicore__ inline void LoadDataWithTranspose(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const LoadData2dTransposeParams& loadDataParams)
        
        II. 参数说明
                            表1 模板参数说明
        参数名                                      描述
        T               Atlas A2 训练系列产品/Atlas 800I A2 推理产品/
                        A200I A2 Box 异构组件，支持的数据类型为：
                        half/bfloat16_t/float/int32_t/uint32_t/uint8_t/int8_t/int4b_t。
                        
                        Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：
                        half/bfloat16_t/float/int32_t/uint32_t/uint8_t/int8_t/int4b_t。
                        
                        Atlas 200I/500 A2 推理产品，支持的数据类型为：
                        uint8_t/int8_t/uint16_t/int16_t/half/bfloat16_t/
                        uint32_t/int32_t/float/int4b_t。

                        其中int4b_t数据类型仅在LocalTensor的TPosition为B2时支持。
        */

        /*
                        表2 参数说明
        参数名称            输入/输出                   含义

        dstLocal            输出            目的操作数，结果矩阵，类型为LocalTensor。
                                            Atlas A2 训练系列产品/Atlas 800I A2 推理产品/
                                            A200I A2 Box 异构组件，支持的TPosition为A2/B2。
                                            
                                            Atlas A3 训练系列产品/Atlas A3 推理系列产品，
                                            支持的TPosition为A2/B2。
                                            
                                            Atlas 200I/500 A2 推理产品，
                                            支持的TPosition为A2/B2。
                                            
                                            LocalTensor的起始地址需要保证512字节对齐。
                                            数据类型和srcLocal的数据类型保持一致。

        srcLocal            输入            源操作数，类型为LocalTensor。
                                            Atlas A2 训练系列产品/Atlas 800I A2 推理产品/
                                            A200I A2 Box 异构组件，支持的TPosition为A1/B1。

                                            Atlas A3 训练系列产品/Atlas A3 推理系列产品，
                                            支持的TPosition为A1/B1。
                                            
                                            Atlas 200I/500 A2 推理产品，
                                            支持的TPosition为A1/B1。
                                            
                                            LocalTensor的起始地址需要保证32字节对齐。
                                            数据类型和dstLocal的数据类型保持一致。
        
        loadDataParams      输入            LoadDataWithTranspose相关参数，
                                            类型为LoadData2dTransposeParams。
                                            具体定义请参考
                                            ${INSTALL_DIR}/include/ascendc/basic_api/interface/kernel_struct_mm.h，
                                            ${INSTALL_DIR}请替换为CANN软件安装后文件存储路径。
                                            参数说明请参考表3。
        */

        /*
                表3 LoadData2dTransposeParams结构体内参数说明
    参数名称                输入/输出                               含义
    startIndex                输入              方块矩阵ID，搬运起始位置为源操作数中第几个方块矩阵
                                                （0 为源操作数中第1个方块矩阵）。
                                                取值范围：startIndex∈[0, 65535] 。默认为0。
                                                例如，源操作数中有20个大小为16*8*4B的分形（数据类型为float），
                                                startIndex=1表示搬运起始位置为第2个方块矩阵，
                                                即将第3和第4个分形从源操作数中转置到目的操作数中
                                                （第1、2个分形组成第1个方块矩阵，第3、4个分形组成第2个方块矩阵）。

    repeatTimes               输入              迭代次数。对于uint8_t/int8_t数据类型，每次迭代处理32*32*1B数据；
                                                对于half/bfloat16_t数据类型，每次迭代处理16*16*2B数据；
                                                对于float/int32_t/uint32_t数据类型，每次迭代处理16*16*4B数据。
                                                对于int4b_t数据类型，每次迭代处理64*64*0.5B数据。
                                                取值范围：repeatTimes∈[0, 255]。默认为0。

    srcStride                 输入              相邻迭代间，源操作数前一个方块矩阵与后一个方块矩阵起始地址的间隔。
                                                这里的单位实际上是拼接后的方块矩阵的大小。
                                                对于uint8_t/int8_t数据类型，单位是32*32*1B；
                                                对于half/bfloat16_t数据类型，单位是16*16*2B；
                                                对于float/int32_t/uint32_t数据类型，单位是16*16*4B。
                                                对于int4b_t数据类型，每次迭代处理64*64*0.5B数据。
                                                取值范围：srcStride∈[0, 65535]。默认为0。

    
    dstGap                     输入             相邻迭代间，目的操作数前一个迭代第一个分形的结束地址到
                                                下一个迭代第一个分形起始地址的间隔，
                                                单位：512B。
                                                取值范围：dstGap∈[0, 65535]。默认为0。

    dstFracGap                 输入             每个迭代内转置的目的操作数前一个分形结束地址与
                                                后一个分形起始地址的间隔，单位为512B，
                                                仅在数据类型为
                                                float/int32_t/uint32_t/uint8_t/int8_t/int4b_t时
                                                有效。取值范围：dstFracGap∈[0, 65535]。默认为0。

addrMode                       输入             预留参数。为后续的功能做保留，开发者暂时无需关注，使用默认值即可。
        */

        /*
        IV. 约束说明
        repeat=0表示不执行搬运操作。
        开发者需要保证目的操作数转置后的分形没有重叠。
        操作数地址偏移对齐要求请参见通用约束。
        */
    }
};

template<class ArchTag>
struct CopyL1ToL0A<
    ArchTag,
    Catlass::Gemm::GemmType<int8_t, Catlass::layout::nZ, AscendC::TPosition::A1>,
    Catlass::Gemm::GemmType<int8_t, Catlass::layout::zZ, AscendC::TPosition::A2>>
{
    using Element = int8_t;
    using LayoutDst = Catlass::layout::zZ;
    using LayoutSrc = Catlass::layout::nZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);

    CATLASS_DEVICE
    CopyL1ToL0A(){}

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::LocalTensor<Element> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        AscendC::LoadData2dTransposeParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)));

        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;

        loadDataParams.dstFracGap = CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)) - 1;

        for (uint32_t i = 0; i < CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(0)); i++) {
            AscendC::LoadDataWithTranspose(dstTensor[i * layoutDst.stride(1) * 2],
                                           srcTensor[i * layoutSrc.stride(1)],
                                           loadDataParams);
        }
    }
};

//////////////////////////////////////////

/// Partial specialization for zN in and zZ out.
template<class ArchTag, class Element>
struct CopyL1ToL0A<ArchTag,
    Catlass::Gemm::GemmType<Element, Catlass::layout::zN, AscendC::TPosition::A1>>
{
    using LayoutDst = Catlass::layout::zZ;
    using LayoutSrc = Catlass::layout::zN;

    static const uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);
    static const uint32_t ELE_NUM_PER_FRACTAL = Catlass::BYTE_PER_FRACTAL / sizeof(Element);

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0A() {};

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

/// Partial specialization for float, zN in and zZ out.
template<class ArchTag>
struct CopyL1ToL0A<ArchTag,
    Catlass::Gemm::GemmType<float,Catlass::layout::zN,AscendC::TPosition::A1>>
{
    using Element = float;
    using LayoutDst = Catlass::layout::zZ;
    using LayoutSrc = Catlass::layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = Catlass::BYTE_PER_FRACTAL / sizeof(Element);

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0A() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        constexpr uint8_t PAD_LIST[4] = {0, 0, 0, 0};
        uint16_t l1M = layoutSrc.shape(0) * layoutSrc.shape(1);
        uint16_t l1K = layoutSrc.shape(2) * layoutSrc.shape(3);
        uint16_t l0M = layoutDst.shape(0) * layoutDst.shape(1);
        uint16_t l0K = layoutDst.shape(2) * layoutDst.shape(3);

        /*
        SetFmatrix
        I. 功能说明
        用于调用Load3Dv1/Load3Dv2时设置FeatureMap的属性描述。
        Load3Dv1/Load3Dv2的模板参数isSetFMatrix设置为false时，
        表示Load3Dv1/Load3Dv2传入的FeatureMap的属性
        （包括l1H、l1W、padList，
        参数介绍参考表4 LoadData3DParamsV1结构体内参数说明、
        表5 LoadData3DParamsV2结构体内参数说明）
        描述不生效，开发者需要通过该接口进行设置。

        II. 函数原型
        __aicore__ inline void SetFmatrix(uint16_t l1H, uint16_t l1W, 
            const uint8_t padList[4], const FmatrixMode& fmatrixMode)
        */

        /*
        III. 参数说明
                    表1 参数说明
        参数名称        输入/输出           含义
        l1H             输入          源操作数height，取值范围：l1H∈[1, 32767]。
        l1W             输入          源操作数width，取值范围：l1W∈[1, 32767] 。
        padList         输入          padding列表 [padding_left, padding_right, padding_top, padding_bottom]，
                                      每个元素取值范围：[0,255]。默认为{0, 0, 0, 0}。

        fmatrixMode     输入          用于控制LoadData指令从left还是right寄存器获取信息。
                                      FmatrixMode类型，定义如下。
                                      当前只支持FMATRIX_LEFT，左右矩阵均使用该配置。
                                      enum class FmatrixMode : uint8_t {
                                        FMATRIX_LEFT = 0,
                                        FMATRIX_RIGHT = 1,
                                    }; 
        
        IV. 约束说明
        该接口需要配合load3Dv1/load3Dv2接口一起使用，需要在load3Dv1/load3Dv2接口之前调用。
        操作数地址偏移对齐要求请参见通用约束。
        */
        AscendC::SetFmatrix(1, l1M, PAD_LIST, AscendC::FmatrixMode::FMATRIX_LEFT);

        static constexpr AscendC::IsResetLoad3dConfig config = {false, false};

        AscendC::LoadData3DParamsV2<Element> loadDataParams;
        loadDataParams.kExtension = l0K;
        loadDataParams.mExtension = l0M;
        loadDataParams.channelSize = l1K;

        AscendC::LoadData<Element, config>(dstTensor, srcTensor, loadDataParams);

        /*
        Load3Dv2接口:
        I. 函数原型：
        template <typename T, const IsResetLoad3dConfig &defaultConfig = IS_RESER_LOAD3D_DEFAULT_CONFIG, 
            typename U = PrimT<T>, typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type = true>
        __aicore__ inline void LoadData(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, 
            const LoadData3DParamsV2<U>& loadDataParams)
        
        1. defaultConfig： 控制是否在Load3Dv1/Load3Dv2接口内部设置相关属性。 
        IsResetLoad3dConfig类型。IsResetLoad3dConfig结构定义如下：
            struct IsResetLoad3dConfig {
                bool isSetFMatrix = true;
                bool isSetPadding = true;
            }; 
        isSetFMatrix配置为true，
        表示在接口内部设置FeatureMap的属性描述
        （包括l1H、l1W、padList，参数介绍参考表4、表5）；设置为false，
        表示该接口传入的FeatureMap的属性描述不生效，开发者需要通过SetFmatrix进行设置。

        isSetPadding配置为true，表示在接口内部设置Pad属性描述
        （即padValue参数，参数介绍参考表4、表5）；
        设置为false，表示该接口传入的Pad属性不生效，
        开发者需要通过SetLoadDataPaddingValue进行设置。可参考样例调用示例。

        该参数的默认值如下：
        constexpr IsResetLoad3dConfig IS_RESER_LOAD3D_DEFAULT_CONFIG = {true, true};

        2. U： LoadData3DParamsV1/LoadData3DParamsV2中padValue的数据类型。
        当dstLocal、srcLocal使用基础数据类型时， U和dstLocal、srcLocal的数据类型T需保持一致，
        否则编译失败。
        当dstLocal 、srcLocal使用TensorTrait类型时，
        U和dstLocal、srcLocal的数据类型T的LiteType需保持一致，否则编译失败。
        最后一个模板参数仅用于上述数据类型检查，用户无需关注。

        II. 参数说明
                    表5 LoadData3DParamsV2结构体内参数说明
    参数名称                                含义
    padList             padding 列表 [padding_left, padding_right, 
                        padding_top, padding_bottom]，
                        每个元素取值范围：[0,255]。默认为{0, 0, 0, 0}。

    l1H                 源操作数height，取值范围：l1H∈[1, 32767]。
    
    l1W                 源操作数weight，取值范围：l1W∈[1, 32767] 。
    
    channelSize         源操作数的通道数，取值范围：channelSize∈[1, 63] 。
                        针对以下型号，channelSize的取值要求为：
                        对于half，channelSize可取值为4，8，16，N * 16 + 4，N * 16 + 8；
                        对于int8_t/uint8_t，channelSize可取值为4，8，16，32，N * 32 + 4，
                        N * 32 + 8，N * 32 + 16。N为正整数。

                        Atlas 推理系列产品 AI Core
                        
                        针对以下型号，channelSize的取值要求为：
                        对于uint32_t/int32_t/float，
                        channelSize可取值为4，N * 8，N * 8 + 4；
                        对于half/bfloat16，
                        channelSize可取值为4，8，N * 16，N * 16 + 4，N * 16 + 8；
                        对于int8_t/uint8_t，
                        channelSize可取值为4，8，16， 32 * N，N * 32 + 4，N * 32 + 8，
                        N * 32 + 16；
                        对于int4b_t，
                        ChannelSize可取值为8，16，32，N * 64，N * 64 + 8，
                        N * 64 + 16，N * 64 + 32。N为正整数。

                        Atlas A2 训练系列产品/Atlas 800I A2 推理产品/
                        A200I A2 Box 异构组件 
                        Atlas A3 训练系列产品/Atlas A3 推理系列产品 
                        Atlas 200I/500 A2 推理产品 

    kExtension          该指令在目的操作数width维度的传输长度，如果不覆盖最右侧的分形，
                        对于half类型，应为16的倍数，对于int8_t/uint8_t应为32的倍数；
                        覆盖的情况则无倍数要求。取值范围: kExtension∈[1, 65535] 。

    mExtension          该指令在目的操作数height维度的传输长度，如果不覆盖最下侧的分形，
                        对于half/int8_t/uint8_t，应为16的倍数；
                        覆盖的情况则无倍数要求。取值范围：mExtension∈[1, 65535] 。

    kStartPt            该指令在目的操作数width维度的起点，对于half类型，应为16的倍数，
                        对于int8_t/uint8_t应为32的倍数。取值范围[0, 65535] 。默认为0。

    mStartPt            该指令在目的操作数height维度的起点，如果不覆盖最下侧的分形，
                        对于half/int8_t/uint8_t，应为16的倍数；
                        覆盖的情况则无倍数要求。取值范围[0, 65535] 。默认为0。

    strideW             卷积核在源操作数width维度滑动的步长，
                        取值范围：strideW∈[1, 63] 。

    strideH             卷积核在源操作数height 维度滑动的步长，
                        取值范围：strideH∈[1, 63] 。

    filterW             卷积核width，取值范围：filterW∈[1, 255] 。

    filterH             卷积核height，取值范围：filterH∈[1, 255] 。

    dilationFilterW     卷积核width膨胀系数，取值范围：dilationFilterW∈[1, 255] 。

    dilationFilterH     卷积核height膨胀系数，取值范围：dilationFilterH∈[1, 255] 。

    enTranspose         是否启用转置功能，对整个目标矩阵进行转置，支持数据类型为 bool，
                        仅在目的TPosition为A2，且源操作数为half类型时有效。默认为false。
                        true：启用
                        false：不启用
    
    enSmallK            是否使能small k特性，每个分形矩阵大小为16*4，支持数据类型为 bool，
                        默认为false。当前产品形态，该特性已不再支持。
                        true：使能
                        false：不使能

    padValue            Pad填充值的数值，数据类型需要与srcLocal保持一致。默认为0。
                        若不想使能padding，可将padList设为全0。

    filterSizeW         是否在filterW的基础上将卷积核width增加256 个元素。true，增加；
                        false，不增加。

    filterSizeH         是否在filterH的基础上将卷积核height增加256个元素。
                        true，增加；false，不增加。

    fMatrixCtrl         表示LoadData3DV2指令从左矩阵还是右矩阵获取FeatureMap的属性描述，
                        与SetFmatrix配合使用，当前只支持设置为false，默认值为false。
                            true：从右矩阵中获取FeatureMap的属性描述；
                            false：从左矩阵中获取FeatureMap的属性描述。
        */
    }
};

template<class ArchTag, class Element>
struct CopyL1ToL0A<ArchTag, 
    Catlass::Gemm::GemmType<Element, Catlass::layout::nZ, AscendC::TPosition::A1>>
{
    using LayoutDst = Catlass::layout::zZ;
    using LayoutSrc = Catlass::layout::nZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = Catlass::BYTE_PER_FRACTAL / sizeof(Element);

    CATLASS_DEVICE
    CopyL1ToL0A() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<Catlass::C0_NUM_PER_FRACTAL>(layoutDst.orgShape(1)));
        loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL;

        loadDataParams.sid = 0;
        loadDataParams.dstGap = layoutDst.stride(3) / ELE_NUM_PER_FRACTAL - 1;
        loadDataParams.ifTranspose = true;
        loadDataParams.addrMode = 0;

        for (uint32_t i = 0; i < CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(0)); i++) {
            AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], 
                srcTensor[i * layoutSrc.stride(1)], 
                loadDataParams);
        }
    }
};

/// Partial specialization for int8_t, nZ in and zZ out. (Transpose A)
template<class ArchTag>
struct CopyL1ToL0A<ArchTag,
    Catlass::Gemm::GemmType<int8_t,Catlass::layout::nZ,AscendC::TPosition::A1>>
{
    using Element = int8_t;
    using LayoutDst = Catlass::layout::zZ;
    using LayoutSrc = Catlass::layout::nZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = Catlass::BYTE_PER_FRACTAL / sizeof(Element);

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0A() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2dTransposeParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)));
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;
        loadDataParams.dstFracGap = CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)) - 1;

        for (uint32_t i = 0; i < CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(0)); i++) {
            AscendC::LoadDataWithTranspose(
                dstTensor[i * layoutDst.stride(1) * 2],
                srcTensor[i * layoutSrc.stride(1)],
                loadDataParams);
        }
    }
};

/// Partial specialization for float, nZ in and zZ out. (Transpose A)
template<class ArchTag>
struct CopyL1ToL0A<ArchTag,
    Catlass::Gemm::GemmType<float, Catlass::layout::nZ, AscendC::TPosition::A1>>
{
    using Element = float;
    using LayoutDst = Catlass::layout::zZ;
    using LayoutSrc = Catlass::layout::nZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = Catlass::BYTE_PER_FRACTAL / sizeof(Element);

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0A() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        constexpr uint8_t PAD_LIST[4] = {0, 0, 0, 0};
        uint16_t l1M = layoutSrc.shape(0) * layoutSrc.shape(1);
        uint16_t l1K = layoutSrc.shape(2) * layoutSrc.shape(3);
        uint16_t l0M = layoutDst.shape(0) * layoutDst.shape(1);
        uint16_t l0K = layoutDst.shape(2) * layoutDst.shape(3);
        // K, M need to be 16 aligned for f32
        uint16_t l1MAlign = RoundUp<Catlass::C0_NUM_PER_FRACTAL>(l1M);
        uint16_t l1KAlign = RoundUp<Catlass::C0_NUM_PER_FRACTAL>(l1K);
        uint16_t l0MAlign = RoundUp<Catlass::C0_NUM_PER_FRACTAL>(l0M);
        uint16_t l0KAlign = RoundUp<Catlass::C0_NUM_PER_FRACTAL>(l0K);
        AscendC::SetFmatrix(1, l1KAlign, PAD_LIST, AscendC::FmatrixMode::FMATRIX_LEFT);
        static constexpr AscendC::IsResetLoad3dConfig config = {false, false};
        AscendC::LoadData3DParamsV2<Element> loadDataParams;
        loadDataParams.kExtension = l0MAlign;
        loadDataParams.mExtension = l0KAlign;
        loadDataParams.enTranspose = true;
        loadDataParams.channelSize = l1MAlign;

        AscendC::LoadData<Element, config>(dstTensor, srcTensor, loadDataParams);
    }
};

///////////////////////////////////////////TileCopyTla//////////////////////////////////////////////////////

/// Partial specialization for CopyL1ToL0A, AtlasA2, zN in and zZ out.
template <class ElementSrc, class ElementDst, class LayoutSrc_, class LayoutDst_>
struct TileCopyTla<Catlass::Arch::AtlasA2, 
    tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc_, AscendC::TPosition::A1>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst_, AscendC::TPosition::A2>,
    std::enable_if_t<tla::detail::iszZ<ElementDst, LayoutDst_>::value &&
                     tla::detail::iszN<ElementSrc, LayoutSrc_>::value>> {
    using LayoutDst = LayoutDst_;
    using LayoutSrc = LayoutSrc_;
    using TensorDst = tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, AscendC::TPosition::A2>;
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

/// Partial specialization for CopyL1ToL0A, AtlasA2, nZ in and zZ out. (Transpose A)
template <class ElementSrc, class ElementDst, class LayoutSrc_, class LayoutDst_>
struct TileCopyTla<Catlass::Arch::AtlasA2, 
    tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc_, AscendC::TPosition::A1>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst_, AscendC::TPosition::A2>,
    std::enable_if_t<tla::detail::iszZ<ElementDst, LayoutDst_>::value &&
                     tla::detail::isnZ<ElementSrc, LayoutSrc_>::value>> {
    using LayoutDst = LayoutDst_;
    using LayoutSrc = LayoutSrc_;
    using TensorDst = tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, AscendC::TPosition::A2>;
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
        const uint32_t dstOuterShapeRow = tla::get<0, 1>(dstTensor.shape());
        const uint32_t dstOuterShapeCol = tla::get<1, 1>(dstTensor.shape());
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());

        AscendC::LoadData2DParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = dstOuterShapeCol;
        loadDataParams.srcStride = 1;
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

/// Partial specialization for CopyL1ToL0A, AtlasA2, int8_t, nZ in and zZ out. (Transpose A)
template <class LayoutSrc_, class LayoutDst_>
struct TileCopyTla<Catlass::Arch::AtlasA2, 
    tla::Tensor<AscendC::LocalTensor<int8_t>, LayoutSrc_, AscendC::TPosition::A1>,
    tla::Tensor<AscendC::LocalTensor<int8_t>, LayoutDst_, AscendC::TPosition::A2>,
    std::enable_if_t<tla::detail::iszZ<int8_t, LayoutDst_>::value &&
                     tla::detail::isnZ<int8_t, LayoutSrc_>::value>> {
    using Element = int8_t;
    using LayoutDst = LayoutDst_;
    using LayoutSrc = LayoutSrc_;
    using TensorDst = tla::Tensor<AscendC::LocalTensor<Element>, LayoutDst, AscendC::TPosition::A2>;
    using TensorSrc = tla::Tensor<AscendC::LocalTensor<Element>, LayoutSrc, AscendC::TPosition::A1>;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = Catlass::BYTE_PER_FRACTAL / sizeof(Element);

    // Mehtods

    CATLASS_DEVICE
    TileCopyTla() {};

    CATLASS_DEVICE
    void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        const uint32_t srcOuterShapeRow = tla::get<0, 1>(srcTensor.shape());
        const uint32_t srcOuterStrideRow = tla::get<0, 1>(srcTensor.stride());
        const uint32_t dstOuterShapeCol = tla::get<1, 1>(dstTensor.shape());
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());

        AscendC::LoadData2dTransposeParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = dstOuterShapeCol;
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;
        loadDataParams.dstFracGap = dstOuterShapeCol - 1;

        for (uint32_t i = 0; i < srcOuterShapeRow; i++) {
            AscendC::LoadDataWithTranspose(dstTensor.data()[i * dstOuterStrideRow * 2],
                                           srcTensor.data()[i * srcOuterStrideRow],
                                           loadDataParams);
        }
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

}

#endif