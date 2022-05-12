
FROM python:3.9.12-bullseye



WORKDIR /usr/src/app
COPY requirements.txt ./
#Install git
RUN apt-get update && \        
     apt-get install -y git

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


RUN apt-get -y update && \
    apt-get install -y \
        automake \
        autoconf \
        gperf \
        build-essential \
        flex \
        bison \
        git && \
    rm -rf /var/lib/apt/lists/*

ARG VERSION=v11_0
ARG JOBS=-j8

RUN git clone http://github.com/steveicarus/iverilog.git
WORKDIR iverilog
RUN git checkout ${VERSION}

ENV JOBS=${JOBS}
ENV VERSION=${VERSION}
ENV IVERILOG_BASE=/tools/eda/iverilog
ENV IVERILOG_MODULEFILES=/tools/modulefiles/eda/iverilog

COPY build.sh ./

RUN ["./build.sh"]


RUN mkdir /code && \      
           cd /code && \        
           git clone https://github.com/NNgen/nngen.git
#Set working directory
WORKDIR /code
RUN python3 nngen/setup.py install 
