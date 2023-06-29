# syntax = docker/dockerfile:1.2
FROM python:3.10

COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt
#RUN --mount=type=cache,target=/var/cache/pip pip install --requirement /tmp/requirements.txt

WORKDIR /app
# Copy all the files of this project inside the container
RUN --mount=type=cache,target=/var/cache/ curl -SL https://minio.lab.sspcloud.fr/cthiounn2/ckpt_dares_ceren_bert_multi.pth -o ckpt_dares_ceren_bert_multi.pth 

COPY . .

CMD ["streamlit", "run", "streamlit-api.py","--server.port", "3838"]

