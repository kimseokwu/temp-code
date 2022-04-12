FROM python:3.7

WORKDIR /usr/src/api

RUN apt-get update && apt-get -y install libgl1

COPY requirements.txt /usr/src/api

RUN pip install --upgrade pip  
RUN pip install torch
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "handler.py"]


