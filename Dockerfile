# Use an official Python runtime as a parent image
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

RUN conda install -yf mkl \
    && conda install -y numpy scipy scikit-learn numexpr "blas=*=*mkl"

# Add unstable repo to allow us to access latest GDAL builds
RUN echo deb http://ftp.uk.debian.org/debian unstable main contrib non-free >> /etc/apt/sources.list
RUN apt-get update

# Existing binutils causes a dependency conflict, correct version will be installed when GDAL gets intalled
RUN apt-get remove -y binutils

# Install GDAL dependencies
RUN apt-get -t unstable install -y libgdal-dev g++

# Update C env vars so compiler can find gdal
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# This will install GDAL 2.2.4
RUN pip install GDAL==3.0.2


# Install any needed packages specified in requirements.txt
COPY requirements.txt /
RUN pip install --trusted-host pypi.python.org -r /requirements.txt

RUN echo Hello there

