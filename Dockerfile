# Используем официальный образ PyTorch с CUDA (или CPU-версию, если GPU нет)
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /workspace

# Копируем файл с зависимостями
COPY requirements.txt .

# Обновляем pip и устанавливаем пакеты
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Копируем весь исходный код, конфиги и утилиты
COPY src/ ./src/
COPY utils/ ./utils/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY *.pkl *.json *.cbm ./
COPY experiments_log.md ./

# Экспонируем порт Streamlit
EXPOSE 8501

# Команда по умолчанию: запуск веб-дашборда
CMD ["streamlit", "run", "src/app.py"]