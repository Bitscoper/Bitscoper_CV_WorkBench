# By Abdullah As-Sadeed

FROM fedora

WORKDIR /application

RUN dnf upgrade -y \
    && dnf install -y python3.11  python3-pip make curl

RUN python3.11 -m ensurepip \
    && pip3.11 install --upgrade pip

COPY ./requirements.txt .
COPY ./Makefile .
RUN make install_dependencies

COPY . .

RUN make update_YOLOv8_default_models

EXPOSE 61117

HEALTHCHECK CMD curl --fail http://localhost:61117/_stcore/health

CMD ["make", "run"]