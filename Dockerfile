# Use the official Python image
FROM python:3.8-slim

# Set environment to production
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn==20.1.0

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Copy the app code
COPY . .

# Expose port 5000
EXPOSE 5000

# Run the application with Gunicorn for production
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
