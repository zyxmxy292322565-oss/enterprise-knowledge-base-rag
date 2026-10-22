FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt redis
# Copy only application source. In particular, do not bake the shared cache or
# environment files into the evaluation image.
COPY *.py /app/
COPY evaluation/evaluation_app.py /app/evaluation_app.py
EXPOSE 8000
CMD ["uvicorn", "api_app:app", "--host", "0.0.0.0", "--port", "8000"]
