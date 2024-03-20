# By Abdullah As-Sadeed

FROM python:3.11

WORKDIR /application

RUN apt update && apt dist-upgrade -y \
    && apt install -y curl make \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt .
COPY ./Makefile .
RUN make install_dependencies

COPY . .

RUN make update_YOLOv8_default_models

EXPOSE 61117

HEALTHCHECK CMD curl --fail http://localhost:61117/_stcore/health

CMD ["make", "run"]