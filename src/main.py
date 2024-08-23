import mlflow


MLFLOW_TRACKING_URI=os.environ['MLFLOW_TRACKING_URI']

mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)

mlflow.set_experiment("Alpha Quickstart")

with mlflow.start_run() as run:
    