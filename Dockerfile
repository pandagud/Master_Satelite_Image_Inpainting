# Use an official Python runtime as a parent image
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

# Install any needed packages specified in requirements.txt
COPY requirements.txt /
RUN pip install --trusted-host pypi.python.org -r /requirements.txt

RUN echo Hello there

