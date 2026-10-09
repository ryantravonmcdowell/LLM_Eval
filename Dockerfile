FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[anthropic]"

# Run as an unprivileged user; outputs go to /work (mount a volume there to keep them).
RUN useradd --create-home --uid 1000 evaluator && mkdir /work && chown evaluator /work
USER evaluator
WORKDIR /work

ENTRYPOINT ["bioeval"]
CMD ["demo", "--out-dir", "/work/out"]
