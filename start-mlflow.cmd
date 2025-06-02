for /f "tokens=1,2 delims==" %%A in (.env) do set %%A=%%B
mlflow server --backend-store-uri $DATABASE --default-artifact-root $S3_PATH --host $HOST --port $PORT
