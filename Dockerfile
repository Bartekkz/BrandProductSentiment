from python:3.6


RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev


WORKDIR /


COPY . /


RUN pip install --upgrade pip


RUN pip3 install -r req.txt


ENTRYPOINT ["python3"]


CMD ["app.py"]
