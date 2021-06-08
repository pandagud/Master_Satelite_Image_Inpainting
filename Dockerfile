# Use an official Python runtime as a parent image
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

# Install any needed packages specified in requirements.txt
COPY requirements.txt /
RUN pip install --trusted-host pypi.python.org -r /requirements.txt
# Install GDAL
RUN conda install --yes --freeze-installed -c conda-forge gdal==3.0.2 \
    && fix-permissions $CONDA_DIR \
    && fix-permissions /home/$NB_USER
RUN echo Hello there

