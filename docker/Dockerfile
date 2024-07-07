# Use an NVIDIA CUDA base image compatible with CUDA 11.0 - most updated tensorflow version in 2.4.0
FROM tensorflow/tensorflow:2.4.0-gpu
ARG DEBIAN_FRONTEND=noninteractive

# Install Python (choose the version you need) and other necessary packages
RUN adduser --system --no-create-home --group --disabled-login --force-badname _apt
RUN chown -R _apt /var/cache/apt/archives/partial
RUN echo 'APT::Get::AllowUnauthenticated "true";' > /etc/apt/apt.conf
RUN echo 'Acquire::Check-Valid-Until false;' >> /etc/apt/apt.conf

# RUN echo "deb http://your.repo.domain/repository/ $(lsb_release -c -s) universe" | tee /etc/apt/sources.list.d/your-repo-name.list

RUN apt -o Acquire::AllowInsecureRepositories=true \
-o Acquire::AllowDowngradeToInsecureRepositories=true \
update

## if the 'apt update' above fails it is likely due to previously
## having the GPG key and repository on the system, you can clean
## out the old lists with `sudo rm /var/lib/apt/lists/your.repo.domain*`

# RUN apt-get -o APT::Get::AllowUnauthenticated=true install repo-keyring-pkgname


# RUN apt-get update
RUN apt-get install -y python3.8 python3-pip
RUN export CUDA_VISIBLE_DEVICES=


# Optionally set up a working directory
WORKDIR /app

# Install additional Python packages
RUN python3.6 -m pip install pygad
RUN python3.6 -m pip install pyldpc
RUN python3.8 -m pip install pexpect
RUN python3.8 -m pip install tqdm
RUN python3.8 -m pip install six
RUN python3.8 -m pip install importlib_metadata


# Copy the rest of your application's code
# COPY . /app

# Set your command or entry point, as appropriate
# CMD ["python3.11", "your_script.py"]