#!/bin/bash

# Read argument
while getopts d: flag
do
    case "${flag}" in
        d) device=${OPTARG};;
    esac
done

# CPU install
if [[ "$device" == "cpu" ]]; then
    echo 'Installing PaddleOCR on CPU'

    python3.9 -m pip install paddleocr paddlepaddle 

    # On Mac M1
    # pip install "paddleocr>=2.0.1" --upgrade PyMuPDF==1.21.1
    # python3 -m pip install paddlepaddle
fi

# GPU install
if [[ "$device" == "gpu" ]]; then
    echo 'Installing PaddleOCR on GPU'

    wget https://developer.download.nvidia.com/compute/redist/cudnn/v8.5.0/local_installers/11.7/cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz

    tar -xvf cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz && \
        cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include && \
        cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64 && \
        chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64

    python3.9 -m pip install paddleocr paddlepaddle-gpu==2.4.2.post117 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html 
fi



