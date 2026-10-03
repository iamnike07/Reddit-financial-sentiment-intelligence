FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download wordnet and vader_lexicon for NLTK during build
RUN python -m nltk.downloader wordnet vader_lexicon punkt

# Copy the rest of the project
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the pipeline then the dashboard
# CMD ["bash", "-c", "python app.py"]
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
