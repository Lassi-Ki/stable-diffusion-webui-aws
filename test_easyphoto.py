import base64
import os
import time
import datetime

import numpy as np
from tqdm import tqdm

import modules.shared as shared
from modules.api import models
from modules.sd_models import reload_model_weights
import traceback
from modules.sd_vae import reload_vae_weights
import json
import requests
from extensions.sd_EasyPhoto.api_test.post_train import post_train
from extensions.sd_EasyPhoto.api_test.post_infer import post_infer
from glob import glob
import cv2
from fastapi import FastAPI


def decode_image_from_base64jpeg(base64_image):
    image_bytes = base64.b64decode(base64_image)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image


def test_invocations(app: FastAPI):
    @app.post("/test_invocations")
    def _test_invocations(req: models.InvocationsRequest):
        print('-------invocation------')
        print(req)
        print("working..........")
        try:
            if req.model != None:
                sd_model_checkpoint = shared.opts.sd_model_checkpoint
                shared.opts.sd_model_checkpoint = req.model
                reload_model_weights()
                if sd_model_checkpoint == shared.opts.sd_model_checkpoint:
                    reload_vae_weights()

            # Train
            if req.s3Url != '':
                shared.download_dataset_from_s3(req.s3Url, f'./datasets/{req.id}')

            img_list = f'./datasets/{req.id}'
            encoded_images = []
            for idx, img_path in enumerate(img_list):
                with open(img_path, "rb") as f:
                    encoded_image = base64.b64encode(f.read()).decode("utf-8")
                    encoded_images.append(encoded_image)
            time_start = time.time()
            outputs = post_train(encoded_images)
            time_end = time.time()
            time_sum = (time_end - time_start) // 60
            print("# --------------------------------------------------------- #")
            print(f"#   Total expenditure：{time_sum} minutes ")
            print("# --------------------------------------------------------- #")
            outputs = json.loads(outputs)
            print(outputs["message"])

            time.sleep(10)

            # Inference
            image_formats = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
            img_list = []
            template_dir = './extensions/sd_EasyPhoto/models/infer_templates/'
            for image_format in image_formats:
                img_list.extend(glob(os.path.join(template_dir, image_format)))
            if len(img_list) == 0:
                print(f" Input template dir {template_dir} contains no images")
            else:
                print(f" Total {len(img_list)} templates to process for {req.id} ID")
            print(img_list)
            now_date = datetime.datetime.now()
            time_start = time.time()
            output_path = './outputs_easyphoto/'

            for img_path in tqdm(img_list):
                print(f" Call generate for ID ({req.id}) and Template ({img_path})")
                with open(img_path, "rb") as f:
                    encoded_image = base64.b64encode(f.read()).decode("utf-8")
                    outputs = post_infer(encoded_image, user_id=req.id)
                    outputs = json.loads(outputs)
                    if len(outputs["outputs"]):
                        image = decode_image_from_base64jpeg(outputs["outputs"][0])
                        toutput_path = os.path.join(os.path.join(output_path),
                                                    f"{req.id}_" + os.path.basename(img_path))
                        print(output_path)
                        cv2.imwrite(toutput_path, image)
                    else:
                        print("Error!", outputs["message"])
                    print(outputs["message"])
            time_end = time.time()
            time_sum = time_end - time_start
            print("# --------------------------------------------------------- #")
            print(f"#   Total expenditure: {time_sum}s")
            print("# --------------------------------------------------------- #")

            # response.images = self.post_invocations(response.images, quality, req.extra_payloads.user_id)
            return outputs

        except Exception as e:
            traceback.print_exc()
            return models.InvocationsErrorResponse(error=str(e))

def post_easyphoto(url="http://0.0.0.0:7860"):
    datas = json.dumps(
        {
            "task": "easyphoto_train",
            "model": "sd_xl_base_1.0.safetensors",
            "id": "tge",
            "s3Url": "s3://sagemaker-us-west-2-011299426194/easyphoto/train_data/user00/",
        }
    )
    print("working...")
    response = requests.post(f"{url}/test_invocations", data=datas, timeout=1500)
    data = response.content.decode("utf-8")
    print(data)
    return data


if __name__ == "__main__":
    try:
        post_easyphoto()
    except Exception as e:
        print(e)
        print("Error: easyphoto_test")
        exit(1)
