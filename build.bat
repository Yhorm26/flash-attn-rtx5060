@echo off
REM 如果编译失败的话，可以参考 : https://blog.csdn.net/qq_42147816/article/details/127160601
REM 编译 CUDA 程序
nvcc -I include src/main.cu -o flash -arch=sm_120a -std=c++17 -w -Xptxas -v,--disable-warnings -lcuda -lcudadevrt

REM 检查编译是否成功
if %errorlevel% equ 0 (
    echo Build succeeded.
) else (
    echo Build failed.
)

echo Start running
flash.exe


pause