FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update

WORKDIR /app_home
COPY *.pickle  ./
COPY *.git  ./
COPY *.py ./
