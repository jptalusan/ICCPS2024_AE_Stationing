FROM python:3.11.7

WORKDIR /usr/src/app

COPY ./environment/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /usr/src/app/code_root/experiments/TEST
CMD ["python", "run_many.py"]
