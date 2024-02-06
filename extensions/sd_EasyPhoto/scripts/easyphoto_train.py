import os
import platform
import subprocess
import sys
import numpy as np
from glob import glob
from shutil import copyfile
from modules.sd_models_config import config_default, config_sdxl
from PIL import Image, ImageOps
from extensions.sd_EasyPhoto.scripts.easyphoto_config import (
    cache_log_file_path,
    easyphoto_models_path,
    models_path,
    scene_id_outpath_samples,
    user_id_outpath_samples,
    validation_prompt,
    validation_prompt_scene,
)
from extensions.sd_EasyPhoto.scripts.easyphoto_utils import (
     check_files_exists_and_download,
     check_id_valid,
     check_scene_valid,
     ep_logger,
     unload_models)
from extensions.sd_EasyPhoto.scripts.sdwebui import get_checkpoint_type, unload_sd
from extensions.sd_EasyPhoto.scripts.train_kohya.utils.lora_utils import convert_lora_to_safetensors

python_executable_path = sys.executable
# base, portrait, sdxl
check_hash = {}


# Attention! Output of js is str or list, not float or int
@unload_sd()
def easyphoto_train_forward(
    sd_model_checkpoint: str,
    id_task: str,
    user_id: str,
    train_mode_choose: str,
    resolution: int,
    val_and_checkpointing_steps: int,
    max_train_steps: int,
    steps_per_photos: int,
    train_batch_size: int,
    gradient_accumulation_steps: int,
    dataloader_num_workers: int,
    learning_rate: float,
    rank: int,
    network_alpha: int,
    validation: bool,
    instance_images: list,
    enable_rl: bool,
    max_rl_time: float,
    timestep_fraction: float,
    skin_retouching_bool: bool,
    training_prefix_prompt: str,
    crop_ratio: float,
    *args,
):
    global check_hash
    print(f'lora training v2: {user_id} start...')

    if user_id == "" or user_id is None:
        ep_logger.error("User id cannot be set to empty.")
        return "User id cannot be set to empty."
    if user_id == "none":
        ep_logger.error("User id cannot be set to none.")
        return "User id cannot be set to none."

    ids = []
    if os.path.exists(user_id_outpath_samples):
        _ids = os.listdir(user_id_outpath_samples)
        for _id in _ids:
            if check_id_valid(_id, user_id_outpath_samples, models_path):
                ids.append(_id)
    ids = sorted(ids)
    if user_id in ids:
        ep_logger.error("User id non-repeatability.")
        return "User id non-repeatability."
    print(f'Now ids: {ids}')
    
    if len(instance_images) == 0:
        ep_logger.error("Please upload training photos.")
        return "Please upload training photos."

    if int(rank) < int(network_alpha):
        ep_logger.error("The network alpha {} must not exceed rank {}. " "It will result in an unintended LoRA.".format(network_alpha, rank))
        return "The network alpha {} must not exceed rank {}. " "It will result in an unintended LoRA.".format(network_alpha, rank)

    check_files_exists_and_download(check_hash.get("base", True), "base")
    check_files_exists_and_download(check_hash.get("portrait", True), "portrait")
    check_hash["base"] = False
    check_hash["portrait"] = False

    checkpoint_type = get_checkpoint_type(sd_model_checkpoint)
    if checkpoint_type == 2:
        ep_logger.error("EasyPhoto does not support the SD2 checkpoint: {}.".format(sd_model_checkpoint))
        return "EasyPhoto does not support the SD2 checkpoint: {}.".format(sd_model_checkpoint)
    sdxl_pipeline_flag = True if checkpoint_type == 3 else False

    if sdxl_pipeline_flag:
        check_files_exists_and_download(check_hash.get("sdxl", True), "sdxl")
        check_hash["sdxl"] = False

    # check if user want to train Scene Lora
    train_scene_lora_bool = True if train_mode_choose == "Train Scene Lora" else False
    if train_scene_lora_bool and float(crop_ratio) < 1:
        ep_logger.warning("The crop ratio {} is smaller than 1. Use original photos to train the scene LoRA.".format(crop_ratio))
    cache_outpath_samples = scene_id_outpath_samples if train_scene_lora_bool else user_id_outpath_samples

    # Check conflicted arguments in SDXL training.
    if sdxl_pipeline_flag:
        if enable_rl:
            ep_logger.error("EasyPhoto does not support RL with the SDXL checkpoint: {}.".format(sd_model_checkpoint))
            return "EasyPhoto does not support RL with the SDXL checkpoint: {}.".format(sd_model_checkpoint)
        if int(resolution) < 1024:
            ep_logger.error("The resolution for SDXL Training needs to be 1024.")
            return "The resolution for SDXL Training needs to be 1024."
        if validation:
            # We do not ensemble models by validation in SDXL training.
            ep_logger.error("To save training time and VRAM, please turn off validation in SDXL training.")
            return "To save training time and VRAM, please turn off validation in SDXL training."

    # Template address
    training_templates_path = os.path.join(easyphoto_models_path, "training_templates")
    # Raw data backup
    original_backup_path = os.path.join(cache_outpath_samples, user_id, "original_backup")
    # Reference backup of face
    if train_scene_lora_bool:
        ref_image_path = os.path.join(models_path, f"Lora/{user_id}.jpg")
    else:
        ref_image_path = os.path.join(cache_outpath_samples, user_id, "ref_image.jpg")

    # Training data retention
    user_path = os.path.join(cache_outpath_samples, user_id, "processed_images")
    images_save_path = os.path.join(cache_outpath_samples, user_id, "processed_images", "train")
    json_save_path = os.path.join(cache_outpath_samples, user_id, "processed_images", "metadata.jsonl")

    # Training weight saving
    weights_save_path = os.path.join(cache_outpath_samples, user_id, "user_weights")
    webui_save_path = os.path.join(models_path, f"Lora/{user_id}.safetensors")
    webui_load_path = os.path.join(models_path, f"Stable-diffusion", sd_model_checkpoint)
    sd_save_path = os.path.join(easyphoto_models_path, "stable-diffusion-v1-5")
    if sdxl_pipeline_flag:
        sd_save_path = sd_save_path.replace("stable-diffusion-v1-5", "stable-diffusion-xl/stabilityai_stable_diffusion_xl_base_1.0")
    if enable_rl and not train_scene_lora_bool:
        ddpo_weight_save_path = os.path.join(cache_outpath_samples, user_id, "ddpo_weights")
        face_lora_path = os.path.join(weights_save_path, f"best_outputs/{user_id}.safetensors")
        ddpo_webui_save_path = os.path.join(models_path, f"Lora/ddpo_{user_id}.safetensors")

    os.makedirs(original_backup_path, exist_ok=True)
    os.makedirs(user_path, exist_ok=True)
    os.makedirs(images_save_path, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(webui_save_path)), exist_ok=True)

    max_train_steps = int(min(len(instance_images) * int(steps_per_photos), int(max_train_steps)))
    local_validation_prompt = None
    if isinstance(instance_images[0], list):
        # User upload training photos with caption files to train the scene lora.
        if train_scene_lora_bool:
            local_validation_prompt = [instance[1] for instance in instance_images]
        instance_images = [instance[0] for instance in instance_images]

    for index, user_image in enumerate(instance_images):
        image = Image.open(user_image["name"])
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.save(os.path.join(original_backup_path, str(index) + ".jpg"))
    
    if local_validation_prompt is None:
        local_validation_prompt = validation_prompt if not train_scene_lora_bool else training_prefix_prompt + ", " + validation_prompt_scene
    # preprocess
    preprocess_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preprocess.py")
    command = [
        f"{python_executable_path}",
        f"{preprocess_path}",
        f"--images_save_path={images_save_path}",
        f"--json_save_path={json_save_path}",
        f"--validation_prompt={local_validation_prompt}",
        f"--inputs_dir={original_backup_path}",
        f"--ref_image_path={ref_image_path}",
        f"--crop_ratio={crop_ratio}",
    ]
    if skin_retouching_bool:
        command += ["--skin_retouching_bool"]
    if train_scene_lora_bool:
        command += ["--train_scene_lora_bool"]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        ep_logger.error(f"Error executing the command: {e}")

    # check preprocess results
    train_images = glob(os.path.join(images_save_path, "*.jpg"))
    if len(train_images) == 0:
        ep_logger.error("Failed to obtain preprocessed images, please check the preprocessing process.")
        return "Failed to obtain preprocessed images, please check the preprocessing process."
    if not os.path.exists(json_save_path):
        ep_logger.error("Failed to obtain preprocessed metadata.jsonl, please check the preprocessing process.")
        return "Failed to obtain preprocessed metadata.jsonl, please check the preprocessing process."

    if not sdxl_pipeline_flag:
        train_kohya_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_kohya/train_lora.py")
    else:
        train_kohya_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_kohya/train_lora_sd_XL.py")
    ep_logger.info("train_file_path : {}".format(train_kohya_path))
    if enable_rl and not train_scene_lora_bool:
        train_ddpo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_kohya/train_ddpo.py")
        ep_logger.info("train_ddpo_path : {}".format(train_kohya_path))

    # outputs/easyphoto-tmp/train_kohya_log.txt, use to cache log and flush to UI
    ep_logger.info("cache_log_file_path: {}".format(cache_log_file_path))
    if not os.path.exists(os.path.dirname(cache_log_file_path)):
        os.makedirs(os.path.dirname(cache_log_file_path), exist_ok=True)

    # Extra arguments to run SDXL training.
    env = None
    if sdxl_pipeline_flag or enable_rl:
        original_config = config_sdxl if sdxl_pipeline_flag else config_default
        sdxl_model_dir = os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models/stable-diffusion-xl")
        pretrained_vae_model_name_or_path = os.path.join(sdxl_model_dir, "madebyollin_sdxl_vae_fp16_fix")
        # SDXL training requires some config files in openai/clip-vit-large-patch14 and laion/CLIP-ViT-bigG-14-laion2B-39B-b160k.
        # We provide them in extensions/EasyPhoto/models. Thus, we need set some environment variables for transformers.
        # if we pass `env` in subprocess.run, the environment variables in the child process will be reset and different from Web UI.
        env = os.environ.copy()
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["TRANSFORMERS_CACHE"] = sdxl_model_dir

    unload_models()
    random_seed = np.random.randint(1, 1e6)
    if platform.system() == "Windows":
        pwd = os.getcwd()
        dataloader_num_workers = 0  # for solve multi process bug
        command = [
            f"{python_executable_path}",
            "-m",
            "accelerate.commands.launch",
            "--mixed_precision=fp16",
            "--main_process_port=3456",
            f"{train_kohya_path}",
            f"--pretrained_model_name_or_path={os.path.relpath(sd_save_path, pwd)}",
            f"--pretrained_model_ckpt={os.path.relpath(webui_load_path, pwd)}",
            f"--train_data_dir={os.path.relpath(user_path, pwd)}",
            "--caption_column=text",
            f"--resolution={resolution}",
            "--random_flip",
            f"--train_batch_size={train_batch_size}",
            f"--gradient_accumulation_steps={gradient_accumulation_steps}",
            f"--dataloader_num_workers={dataloader_num_workers}",
            f"--max_train_steps={max_train_steps}",
            f"--checkpointing_steps={val_and_checkpointing_steps}",
            f"--learning_rate={learning_rate}",
            "--lr_scheduler=constant",
            "--lr_warmup_steps=0",
            "--train_text_encoder",
            f"--seed={random_seed}",
            f"--rank={rank}",
            f"--network_alpha={network_alpha}",
            f"--validation_prompt={local_validation_prompt}",
            f"--validation_steps={val_and_checkpointing_steps}",
            f"--output_dir={os.path.relpath(weights_save_path, pwd)}",
            f"--logging_dir={os.path.relpath(weights_save_path, pwd)}",
            "--enable_xformers_memory_efficient_attention",
            "--mixed_precision=fp16",
            f"--template_dir={os.path.relpath(training_templates_path, pwd)}",
            "--template_mask",
            "--merge_best_lora_based_face_id",
            f"--merge_best_lora_name={user_id}",
            f"--cache_log_file={cache_log_file_path}",
        ]
        if validation and not train_scene_lora_bool:
            command += ["--validation"]
        if sdxl_pipeline_flag:
            command += [f"--original_config={original_config}"]
            command += [f"--pretrained_vae_model_name_or_path={pretrained_vae_model_name_or_path}"]
        if train_scene_lora_bool:
            command += ["--train_scene_lora_bool"]
            # We do not train the text encoders for SDXL Scene Lora.
            if sdxl_pipeline_flag:
                command = [c for c in command if not c.startswith("--train_text_encoder")]
        try:
            subprocess.run(command, env=env, check=True)
        except subprocess.CalledProcessError as e:
            ep_logger.error(f"Error executing the command: {e}")

        # Reinforcement learning after LoRA training.
        if enable_rl and not train_scene_lora_bool:
            # The DDPO (LoRA) distributed training is unstable due to a known accelerate/diffusers issue. Set `num_processes` to 1.
            # See https://github.com/kvablack/ddpo-pytorch/issues/10 for details.
            command = [
                f"{python_executable_path}",
                "-m",
                "accelerate.commands.launch",
                "--mixed_precision=fp16",
                "--main_process_port=4567",
                "--num_processes=1",
                f"{train_ddpo_path}",
                f"--run_name={user_id}",
                f"--logdir={os.path.relpath(ddpo_weight_save_path, pwd)}",
                f"--cache_log_file={cache_log_file_path}",
                f"--pretrained_model_name_or_path={os.path.relpath(sd_save_path, pwd)}",
                f"--pretrained_model_ckpt={os.path.relpath(webui_load_path, pwd)}",
                f"--original_config={original_config}",
                f"--face_lora_path={os.path.relpath(face_lora_path, pwd)}",
                f"--sample_batch_size=4",
                f"--sample_num_batches_per_epoch=2",
                f"--sample_num_steps=50",
                f"--timestep_fraction={timestep_fraction}",
                f"--train_batch_size=1",
                f"--gradient_accumulation_steps=8",
                f"--learning_rate=0.0001",
                f"--seed={random_seed}",
                "--use_lora",
                f"--rank=4",
                f"--cfg",
                f"--allow_tf32",
                f"--num_epochs=200",
                f"--save_freq=1",
                f"--reward_fn=faceid_retina",
                f"--target_image_dir={os.path.relpath(images_save_path, pwd)}",
                f"--per_prompt_stat_tracking",
            ]
            max_rl_time = int(float(max_rl_time) * 60 * 60)
            env["MAX_RL_TIME"] = str(max_rl_time)
            try:
                ep_logger.info("Start RL (reinforcement learning). The max time of RL is {}.".format(max_rl_time))
                # Since `accelerate` spawns a new process, set `timeout` in `subprocess.run` does not take effects.
                subprocess.run(command, env=env, check=True)
            except subprocess.CalledProcessError as e:
                ep_logger.error(f"Error executing the command: {e}")
            finally:
                # The cached log file will be cleared when times out or errors occur.
                with open(cache_log_file_path, "w") as _:
                    pass
    else:
        command = [
            f"{python_executable_path}",
            "-m",
            "accelerate.commands.launch",
            "--mixed_precision=fp16",
            "--main_process_port=3456",
            f"{train_kohya_path}",
            f"--pretrained_model_name_or_path={sd_save_path}",
            f"--pretrained_model_ckpt={webui_load_path}",
            f"--train_data_dir={user_path}",
            "--caption_column=text",
            f"--resolution={resolution}",
            "--random_flip",
            f"--train_batch_size={train_batch_size}",
            f"--gradient_accumulation_steps={gradient_accumulation_steps}",
            f"--dataloader_num_workers={dataloader_num_workers}",
            f"--max_train_steps={max_train_steps}",
            f"--checkpointing_steps={val_and_checkpointing_steps}",
            f"--learning_rate={learning_rate}",
            "--lr_scheduler=constant",
            "--lr_warmup_steps=0",
            "--train_text_encoder",
            f"--seed={random_seed}",
            f"--rank={rank}",
            f"--network_alpha={network_alpha}",
            f"--validation_prompt={local_validation_prompt}",
            f"--validation_steps={val_and_checkpointing_steps}",
            f"--output_dir={weights_save_path}",
            f"--logging_dir={weights_save_path}",
            "--enable_xformers_memory_efficient_attention",
            "--mixed_precision=fp16",
            f"--template_dir={training_templates_path}",
            "--template_mask",
            "--merge_best_lora_based_face_id",
            f"--merge_best_lora_name={user_id}",
            f"--cache_log_file={cache_log_file_path}",
        ]
        if validation and not train_scene_lora_bool:
            command += ["--validation"]
        if sdxl_pipeline_flag:
            command += [f"--original_config={original_config}"]
            command += [f"--pretrained_vae_model_name_or_path={pretrained_vae_model_name_or_path}"]
        if train_scene_lora_bool:
            command += ["--train_scene_lora_bool"]
            # We do not train the text encoders for SDXL Scene Lora.
            if sdxl_pipeline_flag:
                command = [c for c in command if not c.startswith("--train_text_encoder")]
        try:
            subprocess.run(command, env=env, check=True)
        except subprocess.CalledProcessError as e:
            ep_logger.error(f"Error executing the command: {e}")

        # Reinforcement learning after LoRA training.
        if enable_rl and not train_scene_lora_bool:
            # The DDPO (LoRA) distributed training is unstable due to a known accelerate/diffusers issue. Set `num_processes` to 1.
            # See https://github.com/kvablack/ddpo-pytorch/issues/10 for details.
            command = [
                f"{python_executable_path}",
                "-m",
                "accelerate.commands.launch",
                "--mixed_precision=fp16",
                "--main_process_port=4567",
                "--num_processes=1",
                f"{train_ddpo_path}",
                f"--run_name={user_id}",
                f"--logdir={ddpo_weight_save_path}",
                f"--cache_log_file={cache_log_file_path}",
                f"--pretrained_model_name_or_path={sd_save_path}",
                f"--pretrained_model_ckpt={webui_load_path}",
                f"--original_config={original_config}",
                f"--face_lora_path={face_lora_path}",
                f"--sample_batch_size=4",
                f"--sample_num_batches_per_epoch=2",
                f"--sample_num_steps=50",
                f"--timestep_fraction={timestep_fraction}",
                f"--train_batch_size=1",
                f"--gradient_accumulation_steps=8",
                f"--learning_rate=0.0001",
                f"--seed={random_seed}",
                "--use_lora",
                f"--rank=4",
                f"--cfg",
                f"--allow_tf32",
                f"--num_epochs=200",
                f"--save_freq=1",
                f"--reward_fn=faceid_retina",
                f"--target_image_dir={images_save_path}",
                f"--per_prompt_stat_tracking",
            ]
            max_rl_time = int(float(max_rl_time) * 60 * 60)
            env["MAX_RL_TIME"] = str(max_rl_time)
            try:
                ep_logger.info("Start RL (reinforcement learning). The max time of RL is {}.".format(max_rl_time))
                # Since `accelerate` spawns a new process, set `timeout` in `subprocess.run` does not take effects.
                subprocess.run(command, env=env, check=True)
            except subprocess.CalledProcessError as e:
                ep_logger.error(f"Error executing the command: {e}")
            finally:
                # The cached log file will be cleared when times out or errors occur.
                with open(cache_log_file_path, "w") as _:
                    pass

    best_weight_path = os.path.join(weights_save_path, f"best_outputs/{user_id}.safetensors")
    # Currently, SDXL training doesn't support the model selection and ensemble. We use the final
    # trained model as the best for simplicity.
    if sdxl_pipeline_flag:
        best_weight_path = os.path.join(weights_save_path, "pytorch_lora_weights.safetensors")
    if not os.path.exists(best_weight_path):
        return "Failed to obtain Lora after training, please check the training process."

    copyfile(best_weight_path, webui_save_path)

    if enable_rl and not train_scene_lora_bool:
        # Currently, the best (reward_mean) ddpo lora checkpoint will be selected and saved to the WebUI Lora folder.
        best_output_dir = os.path.join(ddpo_weight_save_path, "best_outputs")
        if not os.path.exists(best_output_dir):
            return "Failed to obtain checkpoints after reinforcement learning, please check the training process."
        ddpo_lora_path = os.path.join(best_output_dir, "pytorch_lora_weights.bin")
        convert_lora_to_safetensors(ddpo_lora_path, ddpo_webui_save_path)

    return "The training has been completed."
