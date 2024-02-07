import os
import logging
import requests
import hashlib

from tqdm import tqdm
from modules import shared
from extensions.sd_EasyPhoto.scripts.easyphoto_config import (
    data_path,
    easyphoto_models_path,
    models_path,
    tryon_gallery_dir)
from modelscope.utils.logger import get_logger as ms_get_logger


# Ms logger set
ms_logger = ms_get_logger()
ms_logger.setLevel(logging.ERROR)

# ep logger set
ep_logger_name = "EasyPhoto"
ep_logger = logging.getLogger(ep_logger_name)
ep_logger.propagate = False

for handler in ep_logger.root.handlers:
    if type(handler) is logging.StreamHandler:
        handler.setLevel(logging.ERROR)

stream_handler = logging.StreamHandler()
handlers = [stream_handler]

for handler in handlers:
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(message)s"))
    handler.setLevel("INFO")
    ep_logger.addHandler(handler)

ep_logger.setLevel("INFO")

# download path
controlnet_extensions_path = os.path.join(data_path, "extensions", "sd-webui-controlnet")
controlnet_extensions_builtin_path = os.path.join(data_path, "extensions-builtin", "sd-webui-controlnet")
models_annotator_path = os.path.join(data_path, "models")
if os.path.exists(controlnet_extensions_path):
    controlnet_annotator_cache_path = os.path.join(controlnet_extensions_path, "annotator/downloads/openpose")
    controlnet_cache_path = controlnet_extensions_path
    controlnet_clip_annotator_cache_path = os.path.join(controlnet_extensions_path, "annotator/downloads/clip_vision")
    controlnet_depth_annotator_cache_path = os.path.join(controlnet_extensions_path, "annotator/downloads/midas")
elif os.path.exists(controlnet_extensions_builtin_path):
    controlnet_annotator_cache_path = os.path.join(controlnet_extensions_builtin_path, "annotator/downloads/openpose")
    controlnet_cache_path = controlnet_extensions_builtin_path
    controlnet_clip_annotator_cache_path = os.path.join(controlnet_extensions_builtin_path, "annotator/downloads/clip_vision")
    controlnet_depth_annotator_cache_path = os.path.join(controlnet_extensions_builtin_path, "annotator/downloads/midas")
else:
    controlnet_annotator_cache_path = os.path.join(models_annotator_path, "annotator/downloads/openpose")
    controlnet_cache_path = controlnet_extensions_path
    controlnet_clip_annotator_cache_path = os.path.join(models_annotator_path, "annotator/downloads/clip_vision")
    controlnet_depth_annotator_cache_path = os.path.join(models_annotator_path, "annotator/downloads/midas")

# tryon gallery path
tryon_template_gallery_dir = os.path.join(tryon_gallery_dir, "template")
tryon_cloth_gallery_dir = os.path.join(tryon_gallery_dir, "cloth")

download_urls = {
    # The models are from civitai/6424 & civitai/118913, we saved them to oss for your convenience in downloading the models.
    "base": [
        # base model
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/ChilloutMix-ni-fp16.safetensors",
        # controlnets
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/control_v11p_sd15_canny.pth",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/control_sd15_random_color.pth",
        # vaes
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/vae-ft-mse-840000-ema-pruned.ckpt",
    ],
    "portrait": [
        # controlnet
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/control_v11p_sd15_openpose.pth",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/control_v11f1e_sd15_tile.pth",
        # loras
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/FilmVelvia3.safetensors",
        # controlnet annotator
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/body_pose_model.pth",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/facenet.pth",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/hand_pose_model.pth",
        # other models
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/face_skin.pth",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/face_landmarks.pth",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/makeup_transfer.pth",
        # templates
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/1.jpg",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/2.jpg",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/3.jpg",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/4.jpg",
    ],
    "sdxl": [
        # sdxl
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/diffusers_xl_canny_mid.safetensors",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/thibaud_xl_openpose_256lora.safetensors",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/madebyollin_sdxl_vae_fp16_fix/diffusion_pytorch_model.safetensors",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/madebyollin-sdxl-vae-fp16-fix.safetensors",
    ],
    "add_text2image": [
        # LZ 16k for text2image
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/LZ-16K%2BOptics.safetensors",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/pose_templates/001.png",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/pose_templates/002.png",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/pose_templates/003.png",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/pose_templates/004.png",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/pose_templates/005.png",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/pose_templates/006.png",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/pose_templates/007.png",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/pose_templates/008.png",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/pose_templates/009.png",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/pose_templates/010.png",
    ],
    "add_ipa_base": [
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/ip-adapter-full-face_sd15.pth",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/clip_h.pth",
    ],
    "add_ipa_sdxl": [
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/ip-adapter-plus-face_sdxl_vit-h.safetensors",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/clip_g.pth",
    ],
    "add_video": [
        # new backbone for video
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/majicmixRealistic_v7.safetensors",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/mm_sd_v15_v2.ckpt",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/flownet.pkl",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/dw-ll_ucoco_384.onnx",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/yolox_l.onnx",
    ],
    "add_tryon": [
        # controlnets
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/dpt_hybrid-midas-501f0c75.pt",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/control_v11f1p_sd15_depth.pth",
        # sam
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/sam_vit_l_0b3195.pth",
    ],
    # Scene Lora Collection
    "Christmas_1": [
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/scene_lora/Christmas_1.safetensors",
    ],
    "Cyberpunk_1": [
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/scene_lora/Cyberpunk_1.safetensors",
    ],
    "FairMaidenStyle_1": [
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/scene_lora/FairMaidenStyle_1.safetensors",
    ],
    "Gentleman_1": [
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/scene_lora/Gentleman_1.safetensors",
    ],
    "GuoFeng_1": [
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/scene_lora/GuoFeng_1.safetensors",
    ],
    "GuoFeng_2": [
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/scene_lora/GuoFeng_2.safetensors",
    ],
    "GuoFeng_3": [
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/scene_lora/GuoFeng_3.safetensors",
    ],
    "GuoFeng_4": [
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/scene_lora/GuoFeng_4.safetensors",
    ],
    "Minimalism_1": [
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/scene_lora/Minimalism_1.safetensors",
    ],
    "NaturalWind_1": [
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/scene_lora/NaturalWind_1.safetensors",
    ],
    "Princess_1": [
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/scene_lora/Princess_1.safetensors",
    ],
    "Princess_2": [
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/scene_lora/Princess_2.safetensors",
    ],
    "Princess_3": [
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/scene_lora/Princess_3.safetensors",
    ],
    "SchoolUniform_1": [
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/scene_lora/SchoolUniform_1.safetensors",
    ],
    "SchoolUniform_2": [
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/scene_lora/SchoolUniform_2.safetensors",
    ],
    "lcm": [
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/lcm_lora_sd15.safetensors",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/lcm_lora_sdxl.safetensors",
    ],
    # Tryon Gallery Collections
    # template
    "boy": [
        "https://pai-vision-data-sh.oss-cn-shanghai.aliyuncs.com/aigc-data/easyphoto/tryon/template/boy.jpg",
    ],
    "girl": [
        "https://pai-vision-data-sh.oss-cn-shanghai.aliyuncs.com/aigc-data/easyphoto/tryon/template/girl.jpg",
    ],
    "dress": [
        "https://pai-vision-data-sh.oss-cn-shanghai.aliyuncs.com/aigc-data/easyphoto/tryon/template/dress.jpg",
    ],
    "short": [
        "https://pai-vision-data-sh.oss-cn-shanghai.aliyuncs.com/aigc-data/easyphoto/tryon/template/short.jpg",
    ],
    # cloth
    "demo_white_200": [
        "https://pai-vision-data-sh.oss-cn-shanghai.aliyuncs.com/aigc-data/easyphoto/tryon/cloth/demo_white/demo_white_200.safetensors",
        "https://pai-vision-data-sh.oss-cn-shanghai.aliyuncs.com/aigc-data/easyphoto/tryon/cloth/demo_white/ref_image.jpg",
        "https://pai-vision-data-sh.oss-cn-shanghai.aliyuncs.com/aigc-data/easyphoto/tryon/cloth/demo_white/ref_image_mask.jpg",
    ],
    "demo_black_200": [
        "https://pai-vision-data-sh.oss-cn-shanghai.aliyuncs.com/aigc-data/easyphoto/tryon/cloth/demo_black/demo_black_200.safetensors",
        "https://pai-vision-data-sh.oss-cn-shanghai.aliyuncs.com/aigc-data/easyphoto/tryon/cloth/demo_black/ref_image.jpg",
        "https://pai-vision-data-sh.oss-cn-shanghai.aliyuncs.com/aigc-data/easyphoto/tryon/cloth/demo_black/ref_image_mask.jpg",
    ],
    "demo_purple_200": [
        "https://pai-vision-data-sh.oss-cn-shanghai.aliyuncs.com/aigc-data/easyphoto/tryon/cloth/demo_purple/demo_purple_200.safetensors",
        "https://pai-vision-data-sh.oss-cn-shanghai.aliyuncs.com/aigc-data/easyphoto/tryon/cloth/demo_purple/ref_image.jpg",
        "https://pai-vision-data-sh.oss-cn-shanghai.aliyuncs.com/aigc-data/easyphoto/tryon/cloth/demo_purple/ref_image_mask.jpg",
    ],
    "demo_dress_200": [
        "https://pai-vision-data-sh.oss-cn-shanghai.aliyuncs.com/aigc-data/easyphoto/tryon/cloth/demo_dress/demo_dress_200.safetensors",
        "https://pai-vision-data-sh.oss-cn-shanghai.aliyuncs.com/aigc-data/easyphoto/tryon/cloth/demo_dress/ref_image.jpg",
        "https://pai-vision-data-sh.oss-cn-shanghai.aliyuncs.com/aigc-data/easyphoto/tryon/cloth/demo_dress/ref_image_mask.jpg",
    ],
    "demo_short_200": [
        "https://pai-vision-data-sh.oss-cn-shanghai.aliyuncs.com/aigc-data/easyphoto/tryon/cloth/demo_short/demo_short_200.safetensors",
        "https://pai-vision-data-sh.oss-cn-shanghai.aliyuncs.com/aigc-data/easyphoto/tryon/cloth/demo_short/ref_image.jpg",
        "https://pai-vision-data-sh.oss-cn-shanghai.aliyuncs.com/aigc-data/easyphoto/tryon/cloth/demo_short/ref_image_mask.jpg",
    ],
    "sliders": [
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/sd14_sliders/smiling_sd1_sliders.pt",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/sd14_sliders/age_sd1_sliders.pt",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/xl_sliders/smiling_sdxl_sliders.pt",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/xl_sliders/age_sdxl_sliders.pt",
    ],
}

save_filenames = {
    # The models are from civitai/6424 & civitai/118913, we saved them to oss for your convenience in downloading the models.
    "base": [
        # base model
        os.path.join(models_path, f"Stable-diffusion/Chilloutmix-Ni-pruned-fp16-fix.safetensors"),
        # controlnets
        [
            os.path.join(models_path, f"ControlNet/control_v11p_sd15_canny.pth"),
            os.path.join(controlnet_cache_path, f"models/control_v11p_sd15_canny.pth"),
        ],
        [
            os.path.join(models_path, f"ControlNet/control_sd15_random_color.pth"),
            os.path.join(controlnet_cache_path, f"models/control_sd15_random_color.pth"),
        ],
        # vaes
        os.path.join(models_path, f"VAE/vae-ft-mse-840000-ema-pruned.ckpt"),
    ],
    "portrait": [
        # controlnets
        [
            os.path.join(models_path, f"ControlNet/control_v11p_sd15_openpose.pth"),
            os.path.join(controlnet_cache_path, f"models/control_v11p_sd15_openpose.pth"),
        ],
        [
            os.path.join(models_path, f"ControlNet/control_v11f1e_sd15_tile.pth"),
            os.path.join(controlnet_cache_path, f"models/control_v11f1e_sd15_tile.pth"),
        ],
        # loras
        os.path.join(models_path, f"Lora/FilmVelvia3.safetensors"),
        # controlnet annotator
        os.path.join(controlnet_annotator_cache_path, f"body_pose_model.pth"),
        os.path.join(controlnet_annotator_cache_path, f"facenet.pth"),
        os.path.join(controlnet_annotator_cache_path, f"hand_pose_model.pth"),
        # other models
        os.path.join(easyphoto_models_path, "face_skin.pth"),
        os.path.join(easyphoto_models_path, "face_landmarks.pth"),
        os.path.join(easyphoto_models_path, "makeup_transfer.pth"),
        # templates
        os.path.join(easyphoto_models_path, "training_templates", "1.jpg"),
        os.path.join(easyphoto_models_path, "training_templates", "2.jpg"),
        os.path.join(easyphoto_models_path, "training_templates", "3.jpg"),
        os.path.join(easyphoto_models_path, "training_templates", "4.jpg"),
    ],
    "sdxl": [
        [
            os.path.join(models_path, f"ControlNet/diffusers_xl_canny_mid.safetensors"),
            os.path.join(controlnet_cache_path, f"models/diffusers_xl_canny_mid.safetensors"),
        ],
        [
            os.path.join(models_path, f"ControlNet/thibaud_xl_openpose_256lora.safetensors"),
            os.path.join(controlnet_cache_path, f"models/thibaud_xl_openpose_256lora.safetensors"),
        ],
        os.path.join(easyphoto_models_path, "stable-diffusion-xl/madebyollin_sdxl_vae_fp16_fix/diffusion_pytorch_model.safetensors"),
        os.path.join(models_path, f"VAE/madebyollin-sdxl-vae-fp16-fix.safetensors"),
    ],
    "add_text2image": [
        # sdxl for text2image
        os.path.join(models_path, f"Stable-diffusion/LZ-16K+Optics.safetensors"),
        os.path.join(easyphoto_models_path, "pose_templates", "001.png"),
        os.path.join(easyphoto_models_path, "pose_templates", "002.png"),
        os.path.join(easyphoto_models_path, "pose_templates", "003.png"),
        os.path.join(easyphoto_models_path, "pose_templates", "004.png"),
        os.path.join(easyphoto_models_path, "pose_templates", "005.png"),
        os.path.join(easyphoto_models_path, "pose_templates", "006.png"),
        os.path.join(easyphoto_models_path, "pose_templates", "007.png"),
        os.path.join(easyphoto_models_path, "pose_templates", "008.png"),
        os.path.join(easyphoto_models_path, "pose_templates", "009.png"),
        os.path.join(easyphoto_models_path, "pose_templates", "010.png"),
    ],
    "add_ipa_base": [
        [
            os.path.join(models_path, f"ControlNet/ip-adapter-full-face_sd15.pth"),
            os.path.join(controlnet_cache_path, f"models/ip-adapter-full-face_sd15.pth"),
        ],
        os.path.join(controlnet_clip_annotator_cache_path, f"clip_h.pth"),
    ],
    "add_ipa_sdxl": [
        [
            os.path.join(models_path, f"ControlNet/ip-adapter-plus-face_sdxl_vit-h.safetensors"),
            os.path.join(controlnet_cache_path, f"models/ip-adapter-plus-face_sdxl_vit-h.safetensors"),
        ],
        os.path.join(controlnet_clip_annotator_cache_path, f"clip_g.pth"),
    ],
    "add_video": [
        # new backbone for video
        os.path.join(models_path, f"Stable-diffusion/majicmixRealistic_v7.safetensors"),
        os.path.join(easyphoto_models_path, "mm_sd_v15_v2.ckpt"),
        os.path.join(easyphoto_models_path, "flownet.pkl"),
        os.path.join(controlnet_annotator_cache_path, "dw-ll_ucoco_384.onnx"),
        os.path.join(controlnet_annotator_cache_path, "yolox_l.onnx"),
    ],
    "add_tryon": [
        os.path.join(controlnet_depth_annotator_cache_path, f"dpt_hybrid-midas-501f0c75.pt"),
        os.path.join(models_path, f"ControlNet/control_v11f1p_sd15_depth.pth"),
        os.path.join(easyphoto_models_path, "sam_vit_l_0b3195.pth"),
    ],
    # Scene Lora Collection
    "Christmas_1": [
        os.path.join(models_path, f"Lora/Christmas_1.safetensors"),
    ],
    "Cyberpunk_1": [
        os.path.join(models_path, f"Lora/Cyberpunk_1.safetensors"),
    ],
    "FairMaidenStyle_1": [
        os.path.join(models_path, f"Lora/FairMaidenStyle_1.safetensors"),
    ],
    "Gentleman_1": [
        os.path.join(models_path, f"Lora/Gentleman_1.safetensors"),
    ],
    "GuoFeng_1": [
        os.path.join(models_path, f"Lora/GuoFeng_1.safetensors"),
    ],
    "GuoFeng_2": [
        os.path.join(models_path, f"Lora/GuoFeng_2.safetensors"),
    ],
    "GuoFeng_3": [
        os.path.join(models_path, f"Lora/GuoFeng_3.safetensors"),
    ],
    "GuoFeng_4": [
        os.path.join(models_path, f"Lora/GuoFeng_4.safetensors"),
    ],
    "Minimalism_1": [
        os.path.join(models_path, f"Lora/Minimalism_1.safetensors"),
    ],
    "NaturalWind_1": [
        os.path.join(models_path, f"Lora/NaturalWind_1.safetensors"),
    ],
    "Princess_1": [
        os.path.join(models_path, f"Lora/Princess_1.safetensors"),
    ],
    "Princess_2": [
        os.path.join(models_path, f"Lora/Princess_2.safetensors"),
    ],
    "Princess_3": [
        os.path.join(models_path, f"Lora/Princess_3.safetensors"),
    ],
    "SchoolUniform_1": [
        os.path.join(models_path, f"Lora/SchoolUniform_1.safetensors"),
    ],
    "SchoolUniform_2": [
        os.path.join(models_path, f"Lora/SchoolUniform_2.safetensors"),
    ],
    "lcm": [
        os.path.join(models_path, f"Lora/lcm_lora_sd15.safetensors"),
        os.path.join(models_path, f"Lora/lcm_lora_sdxl.safetensors"),
    ],
    # Tryon Gallery Collections
    # template
    "boy": [os.path.join(tryon_template_gallery_dir, "boy.jpg")],
    "girl": [os.path.join(tryon_template_gallery_dir, "girl.jpg")],
    "dress": [os.path.join(tryon_template_gallery_dir, "dress.jpg")],
    "short": [os.path.join(tryon_template_gallery_dir, "short.jpg")],
    # cloth
    "demo_white_200": [
        os.path.join(models_path, f"Lora/demo_white_200.safetensors"),
        os.path.join(tryon_cloth_gallery_dir, "demo_white_200.jpg"),
        os.path.join(tryon_cloth_gallery_dir, "demo_white_200_mask.jpg"),
    ],
    "demo_black_200": [
        os.path.join(models_path, f"Lora/demo_black_200.safetensors"),
        os.path.join(tryon_cloth_gallery_dir, "demo_black_200.jpg"),
        os.path.join(tryon_cloth_gallery_dir, "demo_black_200_mask.jpg"),
    ],
    "demo_purple_200": [
        os.path.join(models_path, f"Lora/demo_purple_200.safetensors"),
        os.path.join(tryon_cloth_gallery_dir, "demo_purple_200.jpg"),
        os.path.join(tryon_cloth_gallery_dir, "demo_purple_200_mask.jpg"),
    ],
    "demo_dress_200": [
        os.path.join(models_path, f"Lora/demo_dress_200.safetensors"),
        os.path.join(tryon_cloth_gallery_dir, "demo_dress_200.jpg"),
        os.path.join(tryon_cloth_gallery_dir, "demo_dress_200_mask.jpg"),
    ],
    "demo_short_200": [
        os.path.join(models_path, f"Lora/demo_short_200.safetensors"),
        os.path.join(tryon_cloth_gallery_dir, "demo_short_200.jpg"),
        os.path.join(tryon_cloth_gallery_dir, "demo_short_200_mask.jpg"),
    ],
    # Sliders
    "sliders": [
        os.path.join(models_path, f"Lora/smiling_sd1_sliders.pt"),
        os.path.join(models_path, f"Lora/age_sd1_sliders.pt"),
        os.path.join(models_path, f"Lora/smiling_sdxl_sliders.pt"),
        os.path.join(models_path, f"Lora/age_sdxl_sliders.pt"),
    ],
}


def compare_hash_link_file(url, file_path):
    if not shared.opts.data.get("easyphoto_check_hash", True):
        return True
    r = requests.head(url)
    total_size = int(r.headers["Content-Length"])

    res = requests.get(url, stream=True)
    remote_head_hash = hashlib.sha256(res.raw.read(1000)).hexdigest()
    res.close()

    end_pos = total_size - 1000
    headers = {"Range": f"bytes={end_pos}-{total_size-1}"}
    res = requests.get(url, headers=headers, stream=True)
    remote_end_hash = hashlib.sha256(res.content).hexdigest()
    res.close()

    with open(file_path, "rb") as f:
        local_head_data = f.read(1000)
        local_head_hash = hashlib.sha256(local_head_data).hexdigest()

        f.seek(end_pos)
        local_end_data = f.read(1000)
        local_end_hash = hashlib.sha256(local_end_data).hexdigest()

    if remote_head_hash == local_head_hash and remote_end_hash == local_end_hash:
        ep_logger.info(f"{file_path} : Hash match")
        return True

    else:
        ep_logger.info(f" {file_path} : Hash mismatch")
        return False


def urldownload_progressbar(url, file_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(total=total_size, unit="B", unit_scale=True)
    with open(file_path, "wb") as f:
        for chunk in response.iter_content(1024):
            if chunk:
                f.write(chunk)
                progress_bar.update(len(chunk))

    progress_bar.close()


def check_files_exists_and_download(check_hash, download_mode="base"):
    urls, filenames = download_urls[download_mode], save_filenames[download_mode]

    for url, filename in zip(urls, filenames):
        if type(filename) is str:
            filename = [filename]

        exist_flag = False
        for _filename in filename:
            if not check_hash:
                if os.path.exists(_filename):
                    exist_flag = True
                    break
            else:
                if os.path.exists(_filename) and compare_hash_link_file(url, _filename):
                    exist_flag = True
                    break
        if exist_flag:
            continue

        ep_logger.info(f"Start Downloading: {url}")
        os.makedirs(os.path.dirname(filename[0]), exist_ok=True)
        urldownload_progressbar(url, filename[0])


def check_id_valid(user_id, user_id_outpath_samples, models_path):
    face_id_image_path = os.path.join(user_id_outpath_samples, user_id, "ref_image.jpg")
    if not os.path.exists(face_id_image_path):
        return False

    safetensors_lora_path = os.path.join(models_path, "Lora", f"{user_id}.safetensors")
    ckpt_lora_path = os.path.join(models_path, "Lora", f"{user_id}.ckpt")
    if not (os.path.exists(safetensors_lora_path) or os.path.exists(ckpt_lora_path)):
        return False
    return True


def check_scene_valid(lora_path, models_path) -> bool:
    from extensions.sd_EasyPhoto.scripts.sdwebui import read_lora_metadata

    safetensors_lora_path = os.path.join(models_path, "Lora", lora_path)
    if not safetensors_lora_path.endswith("safetensors"):
        return False
    metadata = read_lora_metadata(safetensors_lora_path)
    if str(metadata.get("ep_lora_version", "")).startswith("scene"):
        return True
    else:
        return False