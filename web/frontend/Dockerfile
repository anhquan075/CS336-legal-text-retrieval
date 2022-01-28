FROM python:3.7.4-stretch

WORKDIR /home/user

RUN apt-get update && apt-get install -y curl git pkg-config cmake

# install as a package
COPY requirements.txt /home/user/
RUN pip install -r requirements.txt 

# copy code
COPY . /home/user/
RUN gdown --id 1QMSC_49kOXYb6b64eFKUT_3YBx5OKbvc

EXPOSE 5000

# cmd for running the API
CMD ["streamlit", "run", "webapp.py", "--server.port", "5000"]
