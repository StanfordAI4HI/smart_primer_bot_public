import subprocess
import os


def upload_store():
    result = subprocess.run(
        "aws s3 sync ./store s3://smart-primer-bot/store",
        env=dict(os.environ, AWS_CONFIG_FILE=".aws/config", AWS_SHARED_CREDENTIALS_FILE=".aws/credentials"), shell=True)
    if result.returncode != 0:
        raise ValueError('Store upload failed!')


def download_store():
    result = subprocess.run(
        "aws s3 cp s3://smart-primer-bot/store ./store --recursive",
        env=dict(os.environ, AWS_CONFIG_FILE=".aws/config", AWS_SHARED_CREDENTIALS_FILE=".aws/credentials"), shell=True)
    if result.returncode != 0:
        raise ValueError('Store download failed!')
