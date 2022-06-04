# syntax=docker/dockerfile:1

FROM python:3.9-slim-buster

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY data/ data/

RUN mkdir models