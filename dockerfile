FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependências
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Executar aplicação
COPY ./src ./src

EXPOSE 8000
# CMD ["fastapi", "dev", "src/api.py", "--host", "0.0.0.0"]
CMD ["fastapi", "run", "--workers", "10", "src/api.py", "--host", "0.0.0.0"]
