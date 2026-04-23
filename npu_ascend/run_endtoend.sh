#!/bin/bash
export ASCEND_HOME_PATH=/home/gyh/Ascend/ascend-toolkit/8.2.RC1
export ASCEND_OPP_PATH=$ASCEND_HOME_PATH/opp
export ASCEND_AICPU_PATH=$ASCEND_HOME_PATH
export TOOLCHAIN_HOME=$ASCEND_HOME_PATH/toolkit
export ASCEND_TOOLKIT_HOME=$ASCEND_HOME_PATH
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/lib64/plugin/opskernel:$ASCEND_HOME_PATH/lib64/plugin/nnengine:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
export PATH=$ASCEND_HOME_PATH/bin:$PATH
export PYTHONPATH=$ASCEND_HOME_PATH/python/site-packages:$PYTHONPATH
cd /home/gyh/V-ABFT_TEST
/home/ywc/miniconda3/envs/myLib/bin/python3 fp32_detection_test.py
