FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
RUN apt update && apt install  openssh-server sudo -y
RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1000 test 
RUN  echo 'test:test' | chpasswd
RUN service ssh start
EXPOSE 22

RUN apt-get -y install git
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


RUN curl -fsSL https://code-server.dev/install.sh | sh

RUN code-server --install-extension ms-python.python


WORKDIR /storage/nitzan/code/PatchView
RUN echo 'root:root' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

