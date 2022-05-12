# FROM python:3.7
FROM python:3.8
WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
# RUN cd /workspaces/DeepLearning/
# RUN bs.sh