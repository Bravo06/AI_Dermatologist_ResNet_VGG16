FROM python:3.11.2

RUN apt-get update && apt-get install -y python3-opencv

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install tensorflow==2.13.0
RUN pip install flask==2.2.5
RUN pip install numpy==1.24.2
RUN pip install waitress==3.0.0
RUN pip install opencv-python

EXPOSE 5000

CMD waitress-serve --listen=0.0.0.0:10000 app:app