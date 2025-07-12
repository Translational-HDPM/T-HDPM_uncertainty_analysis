FROM python:3.12.11-bookworm
WORKDIR /usr/local/app

EXPOSE 2025
COPY requirements.txt ./
RUN pip install uv
RUN uv pip install --system --no-cache-dir -r requirements.txt

COPY src ./src
COPY dummy_data ./dummy_data
COPY notebooks/uncertainty_analysis_adjusted_sd.py ./notebooks/uncertainty_analysis_adjusted_sd.py

CMD ["marimo", "run", "--host", "0.0.0.0", "--port", "2025", "--headless", "notebooks/uncertainty_analysis_adjusted_sd.py"]
