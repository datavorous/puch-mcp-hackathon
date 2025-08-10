FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Ensure uv uses a writable cache in /tmp (Spaces runs container as non-root)
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"
ENV HOME=/tmp
ENV UV_CACHE_DIR=/tmp/.cache/uv
# Create and make the cache dir world-writable so any runtime user can write to it
RUN mkdir -p /tmp/.cache/uv && chmod 0777 /tmp/.cache/uv

COPY pyproject.toml uv.lock ./

RUN uv sync --locked --no-cache

COPY . /app

ENV PORT=7860
EXPOSE ${PORT}

CMD ["uv", "run", "src/main.py"]
