# Phase 3: agent replica image. One container = one replica (Agent 1-a/b/c
# in the diagram); uvicorn serves support_agent.api:app on :8000.
# Backend selection is env-driven (see docker-compose.yml); GOOGLE_API_KEY
# is passed through at run time for the live turns.
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY support_agent/ ./support_agent/
COPY rules/ ./rules/
COPY demo.py .

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "support_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
