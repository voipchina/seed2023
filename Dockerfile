FROM cnstark/pytorch:1.13.1-py3.9.16-ubuntu20.04
# FROM cnstark/pytorch:2.0.1-py3.10.11-ubuntu22.04
ENV DEBIAN_FRONTEND noninteractive
WORKDIR /workspace
RUN apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y \
ffmpeg \
libsm6 \
libxext6 \
&& \
apt-get clean && \
rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --progress-bar off --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/
RUN pip install --progress-bar off --no-cache-dir -r requirements.txt  -i https://mirrors.aliyun.com/pypi/simple/
