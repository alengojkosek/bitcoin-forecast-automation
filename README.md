How to Run
To run the project, follow the steps below:

Make sure you have Poetry and npm installed on your system.

Pull the necessary dependencies and data using DVC by running the following command:
poetry run dvc pull -r origin -f

Start the API server by executing the following command:
poetry run src/serve/api.py

Navigate to the client directory by running:
cd src/client/
Start the client application by running the following command:
npm start

This will launch the application and you can access it by opening a web browser and navigating to the specified URL.