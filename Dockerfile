# By Abdullah As-Sadeed

FROM fedora

WORKDIR /application

RUN dnf upgrade -y

COPY ./requirements.txt .
COPY ./Makefile .
RUN make install_dependencies

COPY . .

RUN make update_YOLOv8_default_models

EXPOSE 61117

HEALTHCHECK CMD curl --fail http://localhost:61117/_stcore/health

CMD ["make", "run"]