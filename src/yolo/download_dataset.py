import argparse
import os

from roboflow import Roboflow
from dotenv import load_dotenv

from src.utils import load_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        help="set config file",
        type=str,
        default="configs/params.yaml",
    )
    args = parser.parse_args()
    config = load_config(args.config)["data"]

    load_dotenv()
    rf = Roboflow(api_key=os.environ["ROBOFLOW_KEY"])
    project = rf.workspace(config["workspace"]).project(config["project"])
    version = project.version(config["version"])
    dataset = version.download(config["model_format"], location=config["location"])
