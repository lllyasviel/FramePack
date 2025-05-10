FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel


WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 7860

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN useradd -m -u $USER_ID -g $GROUP_ID -G sudo -s /bin/bash -c '' user
USER user


CMD ["python", "demo_gradio.py", "--server", "0.0.0.0"]