# Use an official Python runtime as a parent image
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

RUN conda install -yf mkl \
    && conda install -y numpy scipy scikit-learn numexpr "blas=*=*mkl"

# Install GDAL
RUN conda install --yes --freeze-installed -c conda-forge gdal==3.0.2 \
    && fix-permissions $CONDA_DIR \
    && fix-permissions /home/$NB_USER

# Clean up after installation
RUN conda clean -afy \
    && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
    && find /opt/conda/ -follow -type f -name '*.js.map' -delete \
    && find /opt/conda/lib/python*/site-packages/bokeh/server/static -follow -type f -name '*.js' ! -name '*.min.js' -delete


# Install any needed packages specified in requirements.txt
COPY requirements.txt /
RUN pip install --trusted-host pypi.python.org -r /requirements.txt

RUN echo Hello there

