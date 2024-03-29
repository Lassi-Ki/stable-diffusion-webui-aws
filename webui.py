from __future__ import annotations

import os
import time

from modules import timer
from modules import initialize_util
from modules import initialize

startup_timer = timer.startup_timer
startup_timer.record("launcher")

initialize.imports()

initialize.check_versions()

from huggingface_hub import hf_hub_download
import boto3
import sys
import json
from modules.sync_models import sync_s3_folder


sys.path.append(os.path.join(os.path.dirname(__file__), 'extensions/sd-webui-controlnet'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'extensions/sd_dreambooth_extension'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'extensions/sd_EasyPhoto'))


def create_api(app):
    from modules.api.api import Api
    from modules.call_queue import queue_lock

    api = Api(app, queue_lock)
    return api


def api_only():
    from fastapi import FastAPI
    from modules.shared_cmd_options import cmd_opts

    initialize.initialize()

    app = FastAPI()
    initialize_util.setup_middleware(app)
    api = create_api(app)

    from modules import script_callbacks
    script_callbacks.before_ui_callback()
    script_callbacks.app_started_callback(None, app)

    print(f"Startup time: {startup_timer.summary()}.")
    api.launch(
        server_name="0.0.0.0" if cmd_opts.listen else "127.0.0.1",
        port=cmd_opts.port if cmd_opts.port else 7861,
        root_path=f"/{cmd_opts.subpath}" if cmd_opts.subpath else ""
    )


def webui():
    from modules.shared_cmd_options import cmd_opts

    launch_api = cmd_opts.api

    if launch_api:
        global cache
        from modules import shared

        # TODO: 将参数里指定的 S3 路径的模型下载到本地
        models_config_s3uri = os.environ.get('models_config_s3uri', None)
        if models_config_s3uri:
            bucket, key = shared.get_bucket_and_key(models_config_s3uri)
            s3_object = shared.s3_client.get_object(Bucket=bucket, Key=key)
            bytes = s3_object["Body"].read()
            payload = bytes.decode('utf8')
            huggingface_models = json.loads(payload).get('huggingface_models', None)
            s3_models = json.loads(payload).get('s3_models', None)
            http_models = json.loads(payload).get('http_models', None)
            s3_embeddings = json.loads(payload).get('s3_embeddings', None)
            s3_vaes = json.loads(payload).get('s3_vaes', None)
        else:
            huggingface_models = os.environ.get('huggingface_models', None)
            huggingface_models = json.loads(huggingface_models) if huggingface_models else None
            s3_models = os.environ.get('s3_models', None)
            s3_models = json.loads(s3_models) if s3_models else None
            http_models = os.environ.get('http_models', None)
            http_models = json.loads(http_models) if http_models else None
            s3_embeddings = os.environ.get('s3_embeddings', None)
            s3_embeddings = json.loads(s3_embeddings) if s3_embeddings else None
            s3_vaes = os.environ.get('s3_vaes', None)
            s3_vaes = json.loads(s3_vaes) if s3_vaes else None
        if huggingface_models:
            for huggingface_model in huggingface_models:
                repo_id = huggingface_model['repo_id']
                filename = huggingface_model['filename']
                name = huggingface_model['name']

                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=f'/tmp/models/{name}',
                    cache_dir='/tmp/cache/huggingface'
                )
        if s3_models:
            for s3_model in s3_models:
                uri = s3_model['uri']
                name = s3_model['name']
                shared.s3_download(uri, f'/tmp/models/{name}')
        if http_models:
            for http_model in http_models:
                uri = http_model['uri']
                filename = http_model['filename']
                name = http_model['name']
                shared.http_download(uri, f'/tmp/models/{name}/{filename}')
        if s3_embeddings:
            print("s3_embeddings:", s3_embeddings)
            for s3_embedding in s3_embeddings:
                uri = s3_embedding['uri']
                name = s3_embedding['name']
                shared.s3_download(uri, f'/tmp/{name}')  # /tmp/embeddings
        if s3_vaes:
            print("s3_vaes:", s3_vaes)
            for s3_vae in s3_vaes:
                uri = s3_vae['uri']
                name = s3_vae['name']
                shared.s3_download(uri, f'/tmp/models/{name}')
        # TODO: 一次性服务器模式时直接将模板和数据集从 S3 下载到本地



        print(os.system('df -h'))
        sd_models_tmp_dir = f"{shared.tmp_models_dir}/Stable-diffusion/"
        cn_models_tmp_dir = f"{shared.tmp_models_dir}/ControlNet/"
        lora_models_tmp_dir = f"{shared.tmp_models_dir}/Lora/"
        cache_dir = f"{shared.tmp_cache_dir}/"
        session = boto3.Session()
        region_name = session.region_name
        sts_client = session.client('sts')
        account_id = sts_client.get_caller_identity()['Account']
        sg_s3_bucket = f"sagemaker-{region_name}-{account_id}"
        if not shared.models_s3_bucket:
            shared.models_s3_bucket = os.environ['sg_default_bucket'] if os.environ.get('sg_default_bucket') else sg_s3_bucket
            shared.s3_folder_sd = "stable-diffusion-webui/models/Stable-diffusion"
            shared.s3_folder_cn = "stable-diffusion-webui/models/ControlNet"
            shared.s3_folder_lora = "stable-diffusion-webui/models/Lora"
        # only download the cn models and the first sd model from default bucket, to accerlate the startup time
        # initial_s3_download(shared.s3_client, shared.s3_folder_sd, sd_models_tmp_dir,cache_dir,'sd')
        sync_s3_folder(sd_models_tmp_dir, cache_dir, 'sd')
        sync_s3_folder(cn_models_tmp_dir, cache_dir, 'cn')
        # sync_s3_folder(lora_models_tmp_dir, cache_dir, 'lora')

    initialize.initialize()

    from modules import shared, ui_tempdir, script_callbacks, ui, progress, ui_extra_networks

    while 1:
        if shared.opts.clean_temp_dir_at_start:
            ui_tempdir.cleanup_tmpdr()
            startup_timer.record("cleanup temp dir")

        script_callbacks.before_ui_callback()
        startup_timer.record("scripts before_ui_callback")

        shared.demo = ui.create_ui()
        startup_timer.record("create ui")

        if not cmd_opts.no_gradio_queue:
            shared.demo.queue(64)

        gradio_auth_creds = list(initialize_util.get_gradio_auth_creds()) or None

        auto_launch_browser = False
        if os.getenv('SD_WEBUI_RESTARTING') != '1':
            if shared.opts.auto_launch_browser == "Remote" or cmd_opts.autolaunch:
                auto_launch_browser = True
            elif shared.opts.auto_launch_browser == "Local":
                auto_launch_browser = not any([cmd_opts.listen, cmd_opts.share, cmd_opts.ngrok, cmd_opts.server_name])

        app, local_url, share_url = shared.demo.launch(
            share=cmd_opts.share,
            server_name=initialize_util.gradio_server_name(),
            server_port=cmd_opts.port,
            ssl_keyfile=cmd_opts.tls_keyfile,
            ssl_certfile=cmd_opts.tls_certfile,
            ssl_verify=cmd_opts.disable_tls_verify,
            debug=cmd_opts.gradio_debug,
            auth=gradio_auth_creds,
            inbrowser=auto_launch_browser,
            prevent_thread_lock=True,
            allowed_paths=cmd_opts.gradio_allowed_path,
            app_kwargs={
                "docs_url": "/docs",
                "redoc_url": "/redoc",
            },
            root_path=f"/{cmd_opts.subpath}" if cmd_opts.subpath else "",
        )

        startup_timer.record("gradio launch")

        app.user_middleware = [x for x in app.user_middleware if x.cls.__name__ != 'CORSMiddleware']

        initialize_util.setup_middleware(app)

        progress.setup_progress_api(app)
        ui.setup_ui_api(app)

        if launch_api:
            create_api(app)

        ui_extra_networks.add_pages_to_demo(app)

        startup_timer.record("add APIs")

        with startup_timer.subcategory("app_started_callback"):
            script_callbacks.app_started_callback(shared.demo, app)

        timer.startup_record = startup_timer.dump()
        print(f"Startup time: {startup_timer.summary()}.")

        try:
            while True:
                server_command = shared.state.wait_for_server_command(timeout=5)
                if server_command:
                    if server_command in ("stop", "restart"):
                        break
                    else:
                        print(f"Unknown server command: {server_command}")
        except KeyboardInterrupt:
            print('Caught KeyboardInterrupt, stopping...')
            server_command = "stop"

        if server_command == "stop":
            print("Stopping server...")
            # If we catch a keyboard interrupt, we want to stop the server and exit.
            shared.demo.close()
            break

        # disable auto launch webui in browser for subsequent UI Reload
        os.environ.setdefault('SD_WEBUI_RESTARTING', '1')

        print('Restarting UI...')
        shared.demo.close()
        time.sleep(0.5)
        startup_timer.reset()
        script_callbacks.app_reload_callback()
        startup_timer.record("app reload callback")
        script_callbacks.script_unloaded_callback()
        startup_timer.record("scripts unloaded callback")
        initialize.initialize_rest(reload_script_modules=True)


if __name__ == "__main__":
    webui()
