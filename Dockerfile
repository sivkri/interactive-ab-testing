# Use the official Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy Streamlit app code
COPY . .

# Expose port Streamlit uses
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "streamlit_AB.py", "--server.port=8501", "--server.address=0.0.0.0"]
