import sys

import gradio as gr

from modules import shared_cmd_options, shared_gradio_themes, options, shared_items, sd_models_types
from modules.paths_internal import models_path, script_path, data_path, sd_configs_path, sd_default_config, sd_model_file, default_sd_model_file, extensions_dir, extensions_builtin_dir  # noqa: F401
from modules import util

cmd_opts = shared_cmd_options.cmd_opts
parser = shared_cmd_options.parser

batch_cond_uncond = True  # old field, unused now in favor of shared.opts.batch_cond_uncond
parallel_processing_allowed = True
styles_filename = cmd_opts.styles_file
config_filename = cmd_opts.ui_settings_file
hide_dirs = {"visible": not cmd_opts.hide_ui_dir_config}

demo = None

device = None

weight_load_location = None

xformers_available = False

hypernetworks = {}

loaded_hypernetworks = []

state = None

prompt_styles = None

interrogator = None

face_restorers = []

options_templates = None
opts = None
restricted_opts = None

sd_model: sd_models_types.WebuiSdModel = None

settings_components = None
"""assinged from ui.py, a mapping on setting names to gradio components repsponsible for those settings"""

tab_names = []

latent_upscale_default_mode = "Latent"
latent_upscale_modes = {
    "Latent": {"mode": "bilinear", "antialias": False},
    "Latent (antialiased)": {"mode": "bilinear", "antialias": True},
    "Latent (bicubic)": {"mode": "bicubic", "antialias": False},
    "Latent (bicubic antialiased)": {"mode": "bicubic", "antialias": True},
    "Latent (nearest)": {"mode": "nearest", "antialias": False},
    "Latent (nearest-exact)": {"mode": "nearest-exact", "antialias": False},
}

sd_upscalers = []

clip_model = None

progress_print_out = sys.stdout

gradio_theme = gr.themes.Base()

total_tqdm = None

mem_mon = None

options_section = options.options_section
OptionInfo = options.OptionInfo
OptionHTML = options.OptionHTML

natural_sort_key = util.natural_sort_key
listfiles = util.listfiles
html_path = util.html_path
html = util.html
walk_files = util.walk_files
ldm_print = util.ldm_print

reload_gradio_theme = shared_gradio_themes.reload_gradio_theme

list_checkpoint_tiles = shared_items.list_checkpoint_tiles
refresh_checkpoints = shared_items.refresh_checkpoints
list_samplers = shared_items.list_samplers
reload_hypernetworks = shared_items.reload_hypernetworks

import threading
models_s3_bucket = None
s3_folder_sd = None
s3_folder_cn = None
s3_folder_lora = None
syncLock = threading.Lock()
tmp_models_dir = '/tmp/models'
tmp_cache_dir = '/tmp/model_sync_cache'
class ModelsRef:
    def __init__(self):
        self.models_ref = {}

    def get_models_ref_dict(self):
        return self.models_ref

    def add_models_ref(self, model_name):
        if model_name in self.models_ref:
            self.models_ref[model_name] += 1
        else:
            self.models_ref[model_name] = 0

    def remove_model_ref(self,model_name):
        if self.models_ref.get(model_name):
            del self.models_ref[model_name]

    def get_models_ref(self, model_name):
        return self.models_ref.get(model_name)

    def get_least_ref_model(self):
        sorted_models = sorted(self.models_ref.items(), key=lambda item: item[1])
        if sorted_models:
            least_ref_model, least_counter = sorted_models[0]
            return least_ref_model,least_counter
        else:
            return None,None

    def pop_least_ref_model(self):
        sorted_models = sorted(self.models_ref.items(), key=lambda item: item[1])
        if sorted_models:
            least_ref_model, least_counter = sorted_models[0]
            del self.models_ref[least_ref_model]
            return least_ref_model,least_counter
        else:
            return None,None

sd_models_Ref = ModelsRef()
cn_models_Ref = ModelsRef()
lora_models_Ref = ModelsRef()

import os
import json
import boto3
import requests
import glob
from botocore.errorfactory import ClientError

cache = dict()
region_name = boto3.session.Session().region_name if not cmd_opts.train else cmd_opts.region_name
s3_client = boto3.client('s3', region_name=region_name)
endpointUrl = s3_client.meta.endpoint_url
s3_client = boto3.client('s3', endpoint_url=endpointUrl, region_name=region_name)
s3_resource= boto3.resource('s3')
generated_images_s3uri = os.environ.get('generated_images_s3uri', None)


def get_bucket_and_key(s3uri):
    pos = s3uri.find('/', 5)
    bucket = s3uri[5 : pos]
    key = s3uri[pos + 1 : ]
    return bucket, key


def download_dataset_from_s3(s3uri, path):
    if path is not None:
        # 如果文件夹不存在就创建它
        if not os.path.exists(path):
            os.makedirs(path)
    pos = s3uri.find('/', 5)
    bucket = s3uri[5: pos]
    key = s3uri[pos + 1:]

    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket)
    for obj in bucket.objects.filter(Prefix=key):
        target = obj.key if path is None else os.path.join(path, os.path.relpath(obj.key, key))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)
    # response = s3.list_objects_v2(Bucket=bucket, Prefix=key)
    # for obj in response.get('Contents', []):
    #     key = obj['Key']
    #     if key.endswith('.jpg') or key.endswith('.png') or key.endswith('.jpeg'):
    #         local_path = os.path.join(path, os.path.basename(key))
    #         s3.download_file(bucket, key, local_path)
    #         print(f'Downloaded {key} to {local_path}')
    #
    # s3 = boto3.resource('s3')
    # bucket = s3.Bucket('sagemaker-us-west-2-011299426194')
    # for obj in bucket.objects.filter(Prefix=s3uri):
    #     if not os.path.exists(os.path.dirname(obj.key)):
    #         os.makedirs(os.path.dirname(obj.key))
    #     bucket.download_file(obj.key, obj.key)


def s3_download(s3uri, path):
    global cache

    print('---path---', path)
    os.system(f'ls -l {os.path.dirname(path)}')

    pos = s3uri.find('/', 5)
    bucket = s3uri[5 : pos]
    key = s3uri[pos + 1 : ]

    objects = []
    paginator = s3_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=key)
    for page in page_iterator:
        if 'Contents' in page:
            for obj in page['Contents']:
                objects.append(obj)
        if 'NextContinuationToken' in page:
            page_iterator = paginator.paginate(Bucket=bucket, Prefix=key,
                                                ContinuationToken=page['NextContinuationToken'])

    try:
        if os.path.isfile('cache'):
            cache = json.load(open('cache', 'r'))
    except:
        pass

    for obj in objects:
        if obj['Size'] == 0:
            continue
        response = s3_client.head_object(
            Bucket = bucket,
            Key =  obj['Key']
        )
        obj_key = 's3://{0}/{1}'.format(bucket, obj['Key'])
        if obj_key not in cache or cache[obj_key] != response['ETag']:
            filename = obj['Key'][obj['Key'].rfind('/') + 1 : ]

            s3_client.download_file(bucket, obj['Key'], os.path.join(path, filename))
            cache[obj_key] = response['ETag']

    json.dump(cache, open('cache', 'w'))

def http_download(httpuri, path):
    with requests.get(httpuri, stream=True) as r:
        r.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def upload_s3files(s3uri, file_path_with_pattern):
    pos = s3uri.find('/', 5)
    bucket = s3uri[5 : pos]
    key = s3uri[pos + 1 : ]

    try:
        for file_path in glob.glob(file_path_with_pattern):
            file_name = os.path.basename(file_path)
            __s3file = f'{key}{file_name}'
            print(file_path, __s3file)
            s3_client.upload_file(file_path, bucket, __s3file)
    except ClientError as e:
        print(e)
        return False
    return True

def upload_s3folder(s3uri, file_path):
    pos = s3uri.find('/', 5)
    bucket = s3uri[5 : pos]
    key = s3uri[pos + 1 : ]

    try:
        for path, _, files in os.walk(file_path):
            for file in files:
                dest_path = path.replace(file_path,"")
                __s3file = f'{key}{dest_path}/{file}'
                __local_file = os.path.join(path, file)
                print(__local_file, __s3file)
                s3_client.upload_file(__local_file, bucket, __s3file)
    except Exception as e:
        print(e)