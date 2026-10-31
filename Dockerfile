FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Las API keys y LLM_PROVIDER se pasan en tiempo de ejecución:
#   docker run --env-file .env -p 8501:8501 <imagen>
EXPOSE 8501

CMD ["streamlit", "run", "app/main_interface.py"]
