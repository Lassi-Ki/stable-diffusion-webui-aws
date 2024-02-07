import json
import requests


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
    response = requests.post(f"{url}/invocations", data=datas, timeout=1500)
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
