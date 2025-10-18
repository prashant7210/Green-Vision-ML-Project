import boto3
import os
from dotenv import load_dotenv
from src.forest.logger import logging

load_dotenv()

class S3Client:
    s3_client = None
    s3_resource = None

    def __init__(self, region_name=os.getenv("AWS_DEFAULT_REGION")):
        if S3Client.s3_client is None or S3Client.s3_resource is None:
            __access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
            __secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

            if __access_key_id is None:
                raise Exception("AWS_ACCESS_KEY_ID is not set in environment variables")
            if __secret_access_key is None:
                raise Exception("AWS_SECRET_ACCESS_KEY is not set in environment variables")

            S3Client.s3_resource = boto3.resource("s3", aws_access_key_id = __access_key_id, aws_secret_access_key = __secret_access_key, region_name = region_name)
            S3Client.s3_client = boto3.client("s3", aws_access_key_id = __access_key_id, aws_secret_access_key = __secret_access_key, region_name = region_name)

            self.s3_client = S3Client.s3_client
            self.s3_resource = S3Client.s3_resource