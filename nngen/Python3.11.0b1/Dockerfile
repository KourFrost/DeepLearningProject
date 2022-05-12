
FROM python:3.7.13-bullseye
WORKDIR /usr/src/app
COPY requirements.txt ./
COPY setup.py ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
# RUN sudo apt install iverilog
RUN python3 setup.py install