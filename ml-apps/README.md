## ML-React-App
It's a template on which we can build a React App and Call Endpoints to Make Predictions.


## Build the Backend (Service)
Move inside the service folder and run the Flask app:

```sh
virtualenv -p Python3 .
source bin/activate
pip install -r requirements.txt
FLASK_APP=app.py flask run
```

## Build the Backend (UI)
Move inside the ui folder and run the UI server app:

```sh
npm install -g serve
npm run build
serve -s build -l 3000
```