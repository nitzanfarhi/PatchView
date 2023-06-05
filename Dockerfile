FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
RUN apt update && apt install  openssh-server sudo -y
RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1000 test 
RUN  echo 'test:test' | chpasswd
RUN service ssh start
EXPOSE 22

RUN apt-get -y install git
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


WORKDIR /app_home

COPY cache_data/orc/* cache_data/orc/
# COPY cache_data/events/gh_cve_proccessed/*.parquet cache_data/events/gh_cve_proccessed/
# COPY cache_data/events/timezones/*  cache_data/events/timezones/
# COPY cache_data/events/*.json cache_data/events/
# COPY cache_data/models cache_data/models
# COPY cache_data/message/* cache_data/message/
# COPY cache_data/code/* cache_data/code/
COPY *.py ./
COPY sweeps/* sweeps/
RUN echo 'root:root' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN service ssh restart
