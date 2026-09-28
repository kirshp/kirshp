# Воспроизводимое окружение без установки Python на хост.
#   docker build -t edu-llm .
#   docker run --rm -it -v "$PWD":/app edu-llm bash
#   (внутри) python prepare_data.py && python train.py && python sample.py
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir torch numpy --index-url https://download.pytorch.org/whl/cpu
COPY . .
CMD ["bash"]
