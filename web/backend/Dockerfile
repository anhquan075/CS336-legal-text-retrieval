FROM nvidia/cuda:11.4.1-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

ARG TAG="hogger32/xlmRoberta-for-VietnameseQA"

#set up environment
RUN apt-get update && apt install -y --no-install-recommends \
    git \
    build-essential \
    python3-pip \
    python3-setuptools \
    unzip

WORKDIR /workingspace

# install python environments 
COPY prepare.sh /workingspace
COPY web/backend/requirements.txt /workingspace/web/backend/
RUN pip3 -q install gdown pip --upgrade \
    && pip3 install -r web/backend/requirements.txt \
    && bash prepare.sh

EXPOSE 5002

# Copy folder into docker
COPY . /workingspace/
WORKDIR /workingspace/web/backend/

# Initialize the QA model
RUN python3 app.py

ENV PYTHONPATH "${PYTHONPATH}:/workingspace/src"

# Cmd for running the API
CMD ["python3", "run.py"]
