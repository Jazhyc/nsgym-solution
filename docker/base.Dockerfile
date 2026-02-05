# Base image for NS-Gym competition -- DO NOT MODIFY
FROM --platform=linux/amd64 python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/
WORKDIR /comp

COPY pyproject.toml ./
RUN uv sync --no-install-project
