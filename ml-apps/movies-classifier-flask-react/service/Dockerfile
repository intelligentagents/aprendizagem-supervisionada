# Use an official Python runtime as a parent image
FROM python:3.7

# Adding backend directory to make absolute filepaths consistent across services
WORKDIR /app/service

# Install Python dependencies
COPY requirements.txt /app/service
RUN pip3 install --upgrade pip -r requirements.txt

# Add the rest of the code
COPY . /app/service

# Make port 5000 available for the app
EXPOSE 5000

# Be sure to use 0.0.0.0 for the host within the Docker container,
# otherwise the browser won't be able to find it
CMD ["python", "app.py"]