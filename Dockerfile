# FROM cnstark/pytorch:1.13.1-py3.9.16-ubuntu20.04
FROM cnstark/pytorch:2.0.1-py3.10.11-ubuntu22.04
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
CMD ["python","main.py"]
