FROM python:3.11
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY . /app
RUN python -m pip cache purge || true
EXPOSE 8501 5000
CMD ["streamlit", "run", "webapp/adaptive_quiz_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
