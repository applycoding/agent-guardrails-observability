FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY agentops ./agentops
RUN pip install --no-cache-dir .
COPY evals ./evals
COPY redteam ./redteam
EXPOSE 8000
CMD ["uvicorn", "agentops.api:app", "--host", "0.0.0.0", "--port", "8000"]
