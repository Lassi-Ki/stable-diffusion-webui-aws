import requests
import json


def post_easyphoto(url="http://0.0.0.0:7860"):
    datas = json.dumps(
        {
            "task": "easyphoto_train",
            "model": "sd_xl_base_1.0.safetensors",
            "id": "tge",
            "s3Url": "s3://sagemaker-us-west-2-011299426194/easyphoto/train_data/user00/",
        }
    )

    response = requests.post(f"{url}/invocations", data=datas, timeout=1500)
    data = response.content.decode("utf-8")
    print(data)
    return data
