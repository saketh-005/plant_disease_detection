# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 7860 available to the world outside this container
EXPOSE 7860

# Define environment variables
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV MPLCONFIGDIR=/tmp/matplotlib

# Create and set permissions for matplotlib config directory
RUN mkdir -p /tmp/matplotlib && \
    chmod -R 777 /tmp/matplotlib

# Run app_resnet9.py when the container launches
CMD ["streamlit", "run", "app_resnet9.py", "--server.port=8501", "--server.address=0.0.0.0"]
