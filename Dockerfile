FROM python:3.7.4

RUN mkdir -p /home/arganbackend
WORKDIR /home/arganbackend

RUN pip install --upgrade pip

ADD requirements.txt .
RUN pip install -r requirements.txt
RUN pip install gunicorn
RUN python -m spacy download en

ADD app app
ADD boot.sh ./
RUN chmod +x boot.sh

ENV FLASK_APP arg_an_backend.py

EXPOSE 5000
ENTRYPOINT ["./boot.sh"]