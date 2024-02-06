import os
from modules.paths import data_path, extensions_builtin_dir, extensions_dir
from modules.shared import tmp_models_dir

# save_dirs
data_dir = data_path
models_path = tmp_models_dir
extensions_builtin_dir = extensions_builtin_dir
extensions_dir = extensions_dir
easyphoto_models_path = os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models")
easyphoto_img2img_samples = os.path.join(data_dir, "outputs/img2img-images")
easyphoto_txt2img_samples = os.path.join(data_dir, "outputs/txt2img-images")
easyphoto_outpath_samples = os.path.join(data_dir, "outputs/easyphoto-outputs")
easyphoto_video_outpath_samples = os.path.join(data_dir, "outputs/easyphoto-video-outputs")
user_id_outpath_samples = os.path.join(data_dir, "outputs/easyphoto-user-id-infos")
cloth_id_outpath_samples = os.path.join(data_dir, "outputs/easyphoto-cloth-id-infos")
scene_id_outpath_samples = os.path.join(data_dir, "outputs/easyphoto-scene-id-infos")
cache_log_file_path = os.path.join(data_dir, "outputs/easyphoto-tmp/train_kohya_log.txt")

# gallery_dir
tryon_preview_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "images"), "tryon")
tryon_gallery_dir = os.path.join(cloth_id_outpath_samples, "gallery")

# prompts
validation_prompt = "easyphoto_face, easyphoto, 1person"
validation_prompt_scene = "special_scene, scene"
validation_tryon_prompt = "easyphoto, 1thing"
DEFAULT_POSITIVE = "(cloth:1.5), (best quality), (realistic, photo-realistic:1.3), (beautiful eyes:1.3), (sparkling eyes:1.3), (beautiful mouth:1.3), finely detail, light smile, extremely detailed CG unity 8k wallpaper, huge filesize, best quality, realistic, photo-realistic, ultra high res, raw photo, put on makeup"
DEFAULT_NEGATIVE = "(bags under the eyes:1.5), (bags under eyes:1.5), (earrings:1.3), (glasses:1.2), (naked:1.5), (nsfw:1.5), nude, breasts, penis, cum, (over red lips: 1.3), (bad lips: 1.3), (bad ears:1.3), (bad hair: 1.3), (bad teeth: 1.3), (worst quality:2), (low quality:2), (normal quality:2), lowres, watermark, badhand, lowres, bad anatomy, bad hands, normal quality, mural,"
DEFAULT_POSITIVE_AD = "(realistic, photorealistic), (masterpiece, best quality, high quality), (delicate eyes and face), extremely detailed CG unity 8k wallpaper, best quality, realistic, photo-realistic, ultra high res, raw photo"
DEFAULT_NEGATIVE_AD = "(naked:1.2), (nsfw:1.2), nipple slip, nude, breasts, (huge breasts:1.2), penis, cum,  (blurry background:1.3), (depth of field:1.7), (holding:2), (worst quality:2), (normal quality:2), lowres, bad anatomy, bad hands"
DEFAULT_POSITIVE_T2I = "(cloth:1.0), (best quality), (realistic, photo-realistic:1.3), film photography, minor acne, (portrait:1.1), (indirect lighting), extremely detailed CG unity 8k wallpaper, huge filesize, best quality, realistic, photo-realistic, ultra high res, raw photo, put on makeup"
DEFAULT_NEGATIVE_T2I = "(nsfw:1.5), (huge breast:1.5), nude, breasts, penis, cum, bokeh, cgi, illustration, cartoon, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, ugly, deformed, blurry, Noisy, log, text (worst quality:2), (low quality:2), (normal quality:2), lowres, watermark, badhand, lowres"

# scene lora
DEFAULT_SCENE_LORA = [
    "Christmas_1",
    "Cyberpunk_1",
    "FairMaidenStyle_1",
    "Gentleman_1",
    "GuoFeng_1",
    "GuoFeng_2",
    "GuoFeng_3",
    "GuoFeng_4",
    "Minimalism_1",
    "NaturalWind_1",
    "Princess_1",
    "Princess_2",
    "Princess_3",
    "SchoolUniform_1",
    "SchoolUniform_2",
]

# tryon template
DEFAULT_TRYON_TEMPLATE = ["boy", "girl", "dress", "short"]

# cloth lora
DEFAULT_CLOTH_LORA = ["demo_black_200", "demo_white_200", "demo_purple_200", "demo_dress_200", "demo_short_200"]

# sliders
DEFAULT_SLIDERS = ["age_sd1_sliders", "smiling_sd1_sliders", "age_sdxl_sliders", "smiling_sdxl_sliders"]

# ModelName
SDXL_MODEL_NAME = "SDXL_1.0_ArienMixXL_v2.0.safetensors"
