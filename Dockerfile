# By Abdullah As-Sadeed

FROM python:3.11-slim

WORKDIR /application

RUN apt update && apt dist-upgrade -y && apt install -y curl make

COPY ./requirements.txt .
COPY ./Makefile .
RUN make install_dependencies

COPY . .

EXPOSE 61117

HEALTHCHECK CMD curl --fail http://localhost:61117/_stcore/health

CMD ["make", "run"]