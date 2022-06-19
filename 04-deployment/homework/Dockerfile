FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

RUN pip install -U pip
RUN pip install pipenv

COPY ["Pipfile", "Pipfile.lock", "starter.py", "./"]

RUN pipenv install --system --deploy

ENTRYPOINT ["python3", "starter.py"]
