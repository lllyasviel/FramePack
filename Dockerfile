FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install sageattention==1.0.6

COPY . .

CMD ["bash"]
