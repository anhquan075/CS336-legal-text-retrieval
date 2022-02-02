FROM nvidia/cuda:11.4.1-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

#set up environment
RUN apt-get update && apt install -y --no-install-recommends \
    tzdata \
    default-jdk \
    git \
    build-essential \
    python3-pip \
    python3-setuptools \
    unzip

WORKDIR /workingspace

# install python environments 
COPY . /workingspace
RUN pip3 -q install gdown pip --upgrade && pip3 install -r web/backend/requirements.txt && bash prepare.sh

EXPOSE 5002

WORKDIR /workingspace/web/backend/

ENV PYTHONPATH "${PYTHONPATH}:/workingspace/src"

# cmd for running the API
CMD ["python3", "run.py"]
