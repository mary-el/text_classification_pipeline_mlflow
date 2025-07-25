## Run Mlflow locally:
* Set environment in `.env`:
  * AWS_ACCESS_KEY_ID
  * AWS_SECRET_ACCESS_KEY
  * MLFLOW_S3_ENDPOINT_URL
  * AWS_S3_DISABLE_CHECKSUM=true
  * S3_PATH
  * DATABASE
  * HOST
  * PORT
```bash
start-mlflow.cmd
```
## Set parameters in config file
### text_classification
* model
  * pretrained_name - hf/local model
  * labels - list of classnames
  * save_to - path to save the best checkpoint
  * onnx_path - path to save model in ONNX format

* data
  * path - path to csv dataframe
  * text_column
  * label_column
  * split_column (optional)
  * other class - "other" class number

* training 
  * learning_rate 
  * batch_size 
  * num_epochs 
  * weight_decay 
  * save_total_limit 
  * output_path - path to checkpoints folder 
  * logs 
  * beta - beta value for f-beta 
  * metric_for_best_model - name of metric to track

* evaluation
  * thresholds - minimal probabilities for each class 
  * results_path - path to save charts and statistics

### yolo
* model
  * name
  * register_name
  * pretrained

* data:
  * workspace
  * project
  * version
  * model_format
  * location

* training: 
  * https://docs.ultralytics.com/modes/train/#train-settings

## Download dataset (for YOLO)
```commandline
python src\yolo\download_dataset.py
```

## Run training pipeline:

```bash
python src\*\main.py
```
```        
usage: main.py [-h] [--eval] [--config CONFIG]

options:
  -h, --help            show this help message and exit
  --eval, -e            evaluate only
  --config CONFIG, -c CONFIG
                        set config file
```
