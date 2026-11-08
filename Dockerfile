FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN useradd --create-home --uid 10001 janus \
    && chown janus:janus /app

COPY --chown=janus:janus pyproject.toml uv.lock README.md ./
COPY --chown=janus:janus config ./config
COPY --chown=janus:janus src ./src
COPY --chmod=755 entrypoint.sh /entrypoint.sh

USER janus

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]

CMD ["fastapi", "run", "--host", "0.0.0.0", "--port", "8000"]
