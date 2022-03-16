FROM python:3

ENV PYTHONUNBUFFERED 1

RUN mkdir /interface

WORKDIR /interface

ADD . /interface

COPY ./requirements.txt /interface/requirements.txt

RUN pip install -r requirements.txt

RUN python newline.py

COPY . /interface/