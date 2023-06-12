from pathlib import Path

import boto3
from botocore.exceptions import ClientError
import os
from midatasets import configs
from loguru import logger
from botocore.config import Config as Boto3Config


class S3Boto3:
    def __init__(
        self,
        endpoint_url: str = configs.aws_endpoint_url,
        config: Boto3Config = Boto3Config(
            retries={"max_attempts": 10, "mode": "adaptive"}
        ),
    ):
        self.s3_resource = boto3.resource(
            "s3", endpoint_url=endpoint_url, config=config
        )
        self.s3_client = boto3.client("s3", endpoint_url=endpoint_url, config=config)

    def check_exists(self, bucket: str, prefix: str):
        try:
            self.s3_client.head_object(Bucket=bucket, Key=prefix)
            return True
        except ClientError:
            return False

    def get_s3_path(self, bucket: str, prefix: str):
        return f"s3://{bucket}/{prefix}"

    def upload_file(self, file_name, bucket, prefix=None, overwrite: bool = False):
        """Upload a file to an S3 bucket

        :param file_name: File to upload
        :param bucket: Bucket to upload to
        :param prefix: S3 object name. If not specified then file_name is used
        :return: True if file was uploaded, else False
        """

        # If S3 prefix was not specified, use file_name
        if prefix is None:
            prefix = os.path.basename(file_name)

        if not overwrite and self.check_exists(bucket, prefix):
            logger.info(
                f"[Upload] {self.get_s3_path(bucket, prefix)} exists -- skipping"
            )
            return True

        try:
            response = self.s3_client.upload_file(file_name, bucket, prefix)
            logger.info(f"[Uploaded] {self.get_s3_path(bucket, prefix)}")
        except ClientError as e:
            logger.error(e)
            return False
        return True

    def download_file(
        self, bucket: str, prefix: str, target: str, overwrite: bool = False
    ):
        s3 = boto3.resource("s3", endpoint_url=configs.aws_endpoint_url)
        bucket = s3.Bucket(bucket)
        target = Path(target)
        if target.exists() and not overwrite:
            logger.info(f"[already exists] {target}, skipping download.")
            return
        if not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"[Downloading] {self.get_s3_path(bucket, prefix)} -> {target}")
        bucket.download_file(prefix, str(target))
