ARG BASE_IMAGE_TYPE=cpu
FROM python:3.8

RUN apt-get update
RUN apt-get install wget ffmpeg libsm6 libxext6  -y

ENV DATA_DIR=/app/FedKNOW/data
RUN mkdir -p $DATA_DIR
ADD data DATA_DIR

WORKDIR /app
# update pip
RUN pip3 install --upgrade pip

# making sure the latest version of flower is installed
RUN pip3 install flwr==0.16.0

ADD docker/requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
# Scripts needed for Flower client
ADD multi /app/FedKNOW/multi
ADD utils /app/FedKNOW/utils
ADD scripts /app/FedKNOW/scripts
ADD data /app/FedKNOW/data
ADD dataset /app/FedKNOW/dataset
ADD models /app/FedKNOW/models
ADD test.py /app/FedKNOW


ENTRYPOINT ["python3","-m","FedKNOW.multi.main_WEIT"]
