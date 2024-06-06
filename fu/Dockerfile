FROM ubuntu:22.04

RUN sed -i 's/http:\/\/archive.ubuntu.com/http:\/\/mirrors.aliyun.com/g' /etc/apt/sources.list && apt-get update && apt-get -y --no-install-recommends install --reinstall ca-certificates &&  sed -i 's/http:\/\/mirrors.aliyun.com/https:\/\/mirrors.aliyun.com/g' /etc/apt/sources.list && apt-get update\
    && apt-get -y --no-install-recommends install python-is-python3 curl git python3 python3-pip ffmpeg libsm6 libxext6 git-lfs \
    && rm -rf /var/lib/apt/lists/* && apt-get clean && pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/

WORKDIR /opt/nuclio/sam

COPY torch-2.3.1+cu118-cp310-cp310-linux_x86_64.whl /tmp 

RUN cd /tmp && pip3 install --no-cache-dir torch-2.3.1+cu118-cp310-cp310-linux_x86_64.whl && rm torch-2.3.1+cu118-cp310-cp310-linux_x86_64.whl

RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && pip3 install --no-cache-dir opencv-python-headless pillow pyyaml pycocotools onnxruntime onnx timm torchpack onnxsim  && pip3 install git+https://gitclone.com/github.com/facebookresearch/segment-anything.git && git lfs install && GIT_LFS_SKIP_SMUDGE=1 git clone https://gitclone.com/github.com/sutungpo/mobileSAMv2.git && cd mobileSAMv2 && pip install -e . && pip3 cache purge

COPY l2.pt mobile_sam.pt /opt/nuclio/sam


