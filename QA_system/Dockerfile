FROM python:3
USER root

# update and install first dependencies
RUN apt-get -y update
RUN apt-get install software-properties-common -y
RUN apt-get install --reinstall ca-certificates

COPY ./QA_system/requirements.txt /requirements.txt
RUN pip install --no-cache-dir --upgrade  -r /requirements.txt


COPY ./QA_system /QA_system
COPY .env .env


# CMD ["python3", "./main.py"]
CMD ["python3", "test_deduplication.py"]