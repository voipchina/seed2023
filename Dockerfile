# FROM cnstark/pytorch:1.13.1-py3.9.16-ubuntu20.04
FROM cnstark/pytorch:2.0.1-py3.10.11-ubuntu22.04
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --progress-bar off --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/
RUN pip install --progress-bar off --no-cache-dir -r requirements.txt  -i https://mirrors.aliyun.com/pypi/simple/
COPY segment.pth .
COPY *.py .
CMD ["python","main.py"]
