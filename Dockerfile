FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
COPY requirements.txt requirements.txt
RUN apt update && apt install  openssh-server sudo -y
RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1000 test 
RUN  echo 'test:test' | chpasswd
RUN service ssh start
EXPOSE 22

RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install -U torch -f https://download.pytorch.org/whl/torch_stable.html 
RUN apt-get update

WORKDIR /app_home
COPY cache_data/models cache_data/models
COPY cache_data/*.json cache_data
COPY cache_data/*.git cache_data
COPY cache_data/*.pickle cache_data
COPY *.py ./

