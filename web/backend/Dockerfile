FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

#set up environment
RUN apt update && apt install -y --no-install-recommends \
    tzdata \
    default-jdk \
    git \
    build-essential \
    python3-pip \
    python3-setuptools \
    unzip

WORKDIR /workingspace

# install python environments 
COPY web/backend/requirements.txt /workingspace
RUN pip3 -q install pip --upgrade && pip3 install -r requirements.txt

EXPOSE 5002

COPY src/ /workingspace/src/
COPY settings/ /workingspace/settings/
COPY data /workingspace/data/
COPY web/backend/ /workingspace/

ENV PYTHONPATH "${PYTHONPATH}:/workingspace/src"

# cmd for running the API
CMD ["python3", "run.py"]
