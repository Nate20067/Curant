FROM python:3.14-slim AS base

#Install OS deps needed for git tooling + PyAudio/ffmpeg build
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        portaudio19-dev \
        libasound-dev \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

#Create virtual environment to mirror local setup
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv "${VIRTUAL_ENV}"

ENV PATH="${VIRTUAL_ENV}/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

WORKDIR /workspace

#Install Python dependencies for the server services
COPY server/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

#Copy server source for iterative dev
COPY server /workspace/server

EXPOSE 8000

CMD ["bash"]
