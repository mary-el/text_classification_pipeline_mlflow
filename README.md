## Run Mlflow locally:
* Set AWS environment:
  * AWS_ACCESS_KEY_ID
  * AWS_SECRET_ACCESS_KEY
  * MLFLOW_S3_ENDPOINT_URL
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```
## Set paramenters in config file
* model
  * pretrained_name - hf/local model
  * labels - list of classnames
* data
  * path - path to csv dataframe

## Run training pipeline:

```bash
python src\main.py
```
```        
usage: main.py [-h] [--eval] [--config CONFIG]

options:
  -h, --help            show this help message and exit
  --eval, -e            evaluate only
  --config CONFIG, -c CONFIG
                        set config file
```
