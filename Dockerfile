# By Abdullah As-Sadeed

FROM python:3.11-slim

WORKDIR /application

RUN apt update && apt dist-upgrade -y && apt install -y curl

COPY ./requirements.txt .
RUN pip3.11 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 61117

HEALTHCHECK CMD curl --fail http://localhost:61117/_stcore/health

CMD ["streamlit", "run", "app.py"]