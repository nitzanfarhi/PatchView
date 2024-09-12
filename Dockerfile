FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel 
RUN apt-get update && \
 apt-get install -y openssh-server
RUN python -m pip install --upgrade pip
RUN python -m pip install transformers tensorboard datasets scikit-learn pandas wandb matplotlib git2json shap torch wandb
# Create an SSH user
RUN useradd -rm -d /home/sshuser -s /bin/bash -g root -G sudo -u 1000 sshuser
# Set the SSH user's password (replace "password" with your desired password)
RUN echo 'sshuser:password' | chpasswd
# Allow SSH access
RUN mkdir /var/run/sshd
# Expose the SSH port
EXPOSE 22
# Start SSH server on container startup
CMD ["/usr/sbin/sshd", "-D"]
