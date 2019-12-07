## ML-React-App
It's a template on which we can build a React App and Call Endpoints to Make Predictions.

## Build images and run containers with docker-compose

First, install the docker and compose to build the images:

**Installing Docker on Windows and Mac:**
Follow the instructions on official documentation to install Docker Desktop: https://docs.docker.com/install/

**Installing Docker on Linux:**
Follow the instructions to Get Docker Engine Community for Ubuntu: https://docs.docker.com/install/linux/docker-ce/ubuntu/ 

**Installing Docker Composer:**
Follow the instructions on docs: https://docs.docker.com/compose/install/

After cloning the repository go into the project folder:

```sh
cd iris-classifier-flask-react
```
Run ```docker-compose up``` which will start a Flask web application for the backend API (default port 5000) and an React frontend web server (default port 3000).


## Building Each Application (Frontend/Backend) Without Docker:

### Build the Backend (Service)
Move inside the service folder and run the Flask app:

```sh
virtualenv -p Python3 .
source bin/activate
pip install -r requirements.txt
FLASK_APP=app.py flask run
```

### Build the Backend (UI)
Move inside the ui folder and run the UI server app:

```sh
npm install -g serve
npm run build
serve -s build -l 3000
```
