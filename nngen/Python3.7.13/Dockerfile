
FROM python:3.7.13-bullseye
WORKDIR /usr/src/app
COPY requirements.txt ./

COPY suggested_setup.py ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
# RUN sudo apt install iverilog

COPY suggested_setup.py ./
RUN python3 suggested_setup.py install

# COPY setup.py ./
# RUN python3 setup.py install