# Use official Python image as base
FROM python:3.12-slim


# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose Streamlit default port
EXPOSE 8501


# Default command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]