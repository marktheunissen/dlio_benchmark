"""
   Copyright (c) 2024, UChicago Argonne, LLC
   All Rights Reserved

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
from time import time

from dlio_benchmark.common.constants import MODULE_STORAGE
from dlio_benchmark.storage.storage_handler import DataStorage, Namespace
from dlio_benchmark.common.enumerations import NamespaceType, MetadataType
import os
import logging
import boto3
import re

from dlio_benchmark.utils.utility import Profile

dlp = Profile(MODULE_STORAGE)

s3 = boto3.resource("s3",
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    endpoint_url=os.environ.get("S3_ENDPOINT"),
    region_name=os.environ.get("AWS_REGION")
)

s3_client = boto3.client("s3",
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    endpoint_url=os.environ.get("S3_ENDPOINT"),
    region_name=os.environ.get("AWS_REGION")
)

class S3Storage(DataStorage):
    """
    Storage APIs for creating files.
    """

    @dlp.log_init
    def __init__(self, namespace, framework=None):
        super().__init__(framework)
        self.namespace = Namespace(namespace, NamespaceType.FLAT)

    @dlp.log
    def get_uri(self, id):
        return "s3://" + os.path.join(self.namespace.name, id)

    @dlp.log
    def create_namespace(self, exist_ok=False):
        return True

    @dlp.log
    def get_namespace(self):
        return self.namespace.name

    @dlp.log
    def create_node(self, id, exist_ok=False):
        return True

    @dlp.log
    def get_node(self, id=""):
        try:
            response = s3_client.head_object(Bucket=self.namespace.name, Key=id)
            return MetadataType.FILE

        except:
            response = s3_client.list_objects_v2(Bucket=self.namespace.name, Prefix=id, Delimiter='/')

            if "Contents" in response or "CommonPrefixes" in response:
                return MetadataType.DIRECTORY
            else:
                return None

    @dlp.log
    def walk_node(self, id, use_pattern=False):
        results=[]
        if use_pattern:
            bucket = s3.Bucket(self.namespace.name)
            pat = re.compile(id)
            for object in bucket.objects.all():
                m = re.search(pat, object.key)
                if m is not None:
                    results.append(self.get_basename(object.key))
        else:
            bucket = s3.Bucket(self.namespace.name)
            for object in bucket.objects.filter(Prefix=id):
                results.append(self.get_basename(object.key))
        return results

    @dlp.log
    def delete_node(self, id):
        bucket = s3.Bucket(self.namespace.name)
        bucket.objects.filter(Prefix=id).delete()
        return True

    @dlp.log
    def put_data(self, id, data, offset=None, length=None):
        s3_client.put_object(Body=data, Bucket=self.namespace.name, Key=id)

    @dlp.log
    def get_data(self, id, data, offset=None, length=None):
        obj=s3_client.get_object(Bucket=self.namespace.name, Key=id)
        return obj["Body"].read()

    def get_basename(self, id):
        return os.path.basename(id)
