## Run Mlflow locally:
* Set environment:
  * AWS_ACCESS_KEY_ID
  * AWS_SECRET_ACCESS_KEY
  * MLFLOW_S3_ENDPOINT_URL
  * AWS_S3_DISABLE_CHECKSUM=true
  * S3_PATH
```bash
start-mlflow.cmd
```
## Set paramenters in config file
* model
  * pretrained_name - hf/local model
  * labels - list of classnames
  * save_to
* data
  * path - path to csv dataframe
  * text_column
  * label_column
  * split_column (optional)

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
