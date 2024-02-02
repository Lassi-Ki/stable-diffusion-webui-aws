import copy
import os
import random
import re
from contextlib import ContextDecorator
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import gradio as gr
import modules
import traceback
import modules.scripts as scripts
import numpy as np
from modules import cache, errors, processing, sd_hijack, sd_models, sd_vae, shared
from modules.paths import models_path
from modules.processing import Processed, StableDiffusionProcessing, StableDiffusionProcessingImg2Img, StableDiffusionProcessingTxt2Img
from modules.sd_models import list_models
from modules.sd_vae import refresh_vae_list
from modules.shared import opts
from modules.timer import Timer

from scripts.easyphoto_utils import (
    AnimateDiffControl,
    AnimateDiffI2VLatent,
    AnimateDiffInfV2V,
    AnimateDiffLora,
    AnimateDiffOutput,
    AnimateDiffProcess,
    AnimateDiffPromptSchedule,
    AnimateDiffUiGroup,
    ep_logger,
    motion_module,
    update_infotext,
    video_visible,
)

output_pic_dir = os.path.join(os.path.dirname(__file__), "online_files/output")

InputImage = Union[np.ndarray, str]
InputImage = Union[Dict[str, InputImage], Tuple[InputImage, InputImage], InputImage]


class unload_sd(ContextDecorator):
    """Context-manager that unloads SD checkpoint to free VRAM."""

    def __enter__(self):
        sd_models.unload_model_weights()

    def __exit__(self, exc_type, exc_val, exc_tb):
        sd_models.reload_model_weights()
        sd_hijack.apply_optimizations()  # cross attention optimization.


class switch_sd_model_vae(ContextDecorator):
    """Context-manager that supports switch SD checkpoint and VAE."""

    def __enter__(self):
        self.origin_sd_model_checkpoint = shared.opts.sd_model_checkpoint
        self.origin_sd_vae = shared.opts.sd_vae
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        shared.opts.sd_model_checkpoint = self.origin_sd_model_checkpoint
        # SD Web UI will check self.origin_sd_model_checkpoint == shared.opts.sd_model_checkpoint automatically.
        sd_models.reload_model_weights()
        shared.opts.sd_vae = self.origin_sd_vae
        # SD Web UI will check self.origin_sd_vae == shared.opts.sd_vae automatically.
        sd_vae.reload_vae_weights()


class switch_return_opt(ContextDecorator):
    """Context-manager that supports switch `return_mask` and `return_mask_composite`."""

    def __enter__(self):
        self.origin_return_mask = shared.opts.return_mask
        self.origin_return_mask_composite = shared.opts.return_mask_composite
        self.origin_control_net_no_detectmap = shared.opts.data.get("control_net_no_detectmap", False)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        shared.opts.return_mask = self.origin_return_mask
        shared.opts.return_mask_composite = self.origin_return_mask_composite
        if "control_net_no_detectmap" in shared.opts.data.keys():
            shared.opts.data["control_net_no_detectmap"] = self.origin_control_net_no_detectmap


class ControlMode(Enum):
    """
    The improved guess mode.
    """

    BALANCED = "Balanced"
    PROMPT = "My prompt is more important"
    CONTROL = "ControlNet is more important"


class ResizeMode(Enum):
    """
    Resize modes for ControlNet input images.
    """

    RESIZE = "Just Resize"
    INNER_FIT = "Crop and Resize"
    OUTER_FIT = "Resize and Fill"

    def int_value(self):
        if self == ResizeMode.RESIZE:
            return 0
        elif self == ResizeMode.INNER_FIT:
            return 1
        elif self == ResizeMode.OUTER_FIT:
            return 2
        assert False, "NOTREACHED"


class ControlNetUnit:
    """
    Represents an entire ControlNet processing unit.
    """

    def __init__(
        self,
        enabled: bool = True,
        module: Optional[str] = None,
        model: Optional[str] = None,
        weight: float = 1.0,
        image: Optional[InputImage] = None,
        resize_mode: Union[int, str] = 1,
        low_vram: bool = False,
        processor_res: int = -1,
        threshold_a: float = -1,
        threshold_b: float = -1,
        guidance_start: float = 0.0,
        guidance_end: float = 1.0,
        pixel_perfect: bool = False,
        control_mode: Union[int, str] = 0,
        save_detected_map: bool = True,
        batch_images=[],
        **_kwargs,
    ):
        self.enabled = enabled
        self.module = module
        self.model = model
        self.weight = weight
        self.image = image
        self.resize_mode = resize_mode
        self.low_vram = low_vram
        self.processor_res = processor_res
        self.threshold_a = threshold_a
        self.threshold_b = threshold_b
        self.guidance_start = guidance_start
        self.guidance_end = guidance_end
        self.pixel_perfect = pixel_perfect
        self.control_mode = control_mode
        self.save_detected_map = save_detected_map
        self.batch_images = batch_images

    def __eq__(self, other):
        if not isinstance(other, ControlNetUnit):
            return False

        return vars(self) == vars(other)


def find_cn_script(script_runner: scripts.ScriptRunner) -> Optional[scripts.Script]:
    """
    Find the ControlNet script in `script_runner`. Returns `None` if `script_runner` does not contain a ControlNet script.
    """

    if script_runner is None:
        return None

    for script in script_runner.alwayson_scripts:
        if is_cn_script(script):
            return script


def update_cn_script_in_place(
    script_runner: scripts.ScriptRunner,
    script_args: List[Any],
    cn_units: List[ControlNetUnit],
    **_kwargs,  # for backwards compatibility
):
    """
    Update the arguments of the ControlNet script in `script_args` in place, reading from `cn_units`.
    `cn_units` and its elements are not modified. You can call this function repeatedly, as many times as you want.

    Does not update `script_args` if any of the folling is true:
    - ControlNet is not present in `script_runner`
    - `script_args` is not filled with script arguments for scripts that are processed before ControlNet
    """

    cn_script = find_cn_script(script_runner)
    if cn_script is None or len(script_args) < cn_script.args_from:
        return

    # fill in remaining parameters to satisfy max models, just in case script needs it.
    max_models = shared.opts.data.get("control_net_max_models_num", 1)
    cn_units = cn_units + [ControlNetUnit(enabled=False)] * max(max_models - len(cn_units), 0)

    cn_script_args_diff = 0
    for script in script_runner.alwayson_scripts:
        if script is cn_script:
            cn_script_args_diff = len(cn_units) - (cn_script.args_to - cn_script.args_from)
            script_args[script.args_from : script.args_to] = cn_units
            script.args_to = script.args_from + len(cn_units)
        else:
            script.args_from += cn_script_args_diff
            script.args_to += cn_script_args_diff


def update_cn_script_in_processing(
    p: processing.StableDiffusionProcessing,
    cn_units: List[ControlNetUnit],
    **_kwargs,  # for backwards compatibility
):
    """
    Update the arguments of the ControlNet script in `p.script_args` in place, reading from `cn_units`.
    `cn_units` and its elements are not modified. You can call this function repeatedly, as many times as you want.

    Does not update `p.script_args` if any of the folling is true:
    - ControlNet is not present in `p.scripts`
    - `p.script_args` is not filled with script arguments for scripts that are processed before ControlNet
    """

    cn_units_type = type(cn_units) if type(cn_units) in (list, tuple) else list
    script_args = list(p.script_args)
    update_cn_script_in_place(p.scripts, script_args, cn_units)
    p.script_args = cn_units_type(script_args)


def is_cn_script(script: scripts.Script) -> bool:
    """
    Determine whether `script` is a ControlNet script.
    """

    return script.title().lower() == "controlnet"


def init_default_script_args(script_runner):
    # find max idx from the scripts in runner and generate a none array to init script_args
    last_arg_index = 1
    for script in script_runner.scripts:
        if last_arg_index < script.args_to:
            last_arg_index = script.args_to
    # None everywhere except position 0 to initialize script args
    script_args = [None] * last_arg_index
    script_args[0] = 0

    # get default values
    with gr.Blocks():  # will throw errors calling ui function without this
        for script in script_runner.scripts:
            if script.ui(script.is_img2img):
                ui_default_values = []
                for elem in script.ui(script.is_img2img):
                    ui_default_values.append(elem.value)
                script_args[script.args_from : script.args_to] = ui_default_values
    return script_args


# Animatediff is not Support when stable-diffusion webui is under v1.6.0.
if video_visible:

    class AnimateDiffScript(scripts.Script):
        def __init__(self):
            self.lora_hacker = None
            self.cfg_hacker = None
            self.cn_hacker = None
            self.prompt_scheduler = None
            self.hacked = False
            self.name = self.title()
            print("AnimateDiffScript init")

        def title(self):
            return "animatediff_easyphoto"

        def show(self, is_img2img):
            return scripts.AlwaysVisible

        def ui(self, is_img2img):
            return (AnimateDiffUiGroup().render(),)

        def before_process(self, p: StableDiffusionProcessing, params: AnimateDiffProcess):
            if p.is_api and isinstance(params, dict):
                self.ad_params = AnimateDiffProcess(**params)
                params = self.ad_params
            if params.enable:
                if self.hacked:
                    self.cn_hacker.restore()
                    self.cfg_hacker.restore()
                    self.lora_hacker.restore()
                    motion_module.restore(p.sd_model)
                    self.hacked = False
                ep_logger.info("AnimateDiff process start.")
                params.set_p(p)
                motion_module.inject(p.sd_model, params.model)
                self.prompt_scheduler = AnimateDiffPromptSchedule()
                self.lora_hacker = AnimateDiffLora(motion_module.mm.is_v2)
                self.lora_hacker.hack()
                self.cfg_hacker = AnimateDiffInfV2V(p, self.prompt_scheduler)
                self.cfg_hacker.hack(params)
                self.cn_hacker = AnimateDiffControl(p, self.prompt_scheduler)
                self.cn_hacker.hack(params)
                update_infotext(p, params)
                self.hacked = True
            elif self.hacked:
                self.cn_hacker.restore()
                self.cfg_hacker.restore()
                self.lora_hacker.restore()
                motion_module.restore(p.sd_model)
                self.hacked = False

        def before_process_batch(self, p: StableDiffusionProcessing, params: AnimateDiffProcess, **kwargs):
            if p.is_api and isinstance(params, dict):
                params = self.ad_params
            if params.enable and isinstance(p, StableDiffusionProcessingImg2Img) and not hasattr(p, "_animatediff_i2i_batch"):
                AnimateDiffI2VLatent().randomize(p, params)

        def postprocess(self, p: StableDiffusionProcessing, res: Processed, params: AnimateDiffProcess):
            if p.is_api and isinstance(params, dict):
                params = self.ad_params
            if params.enable:
                self.prompt_scheduler.save_infotext_txt(res)
                self.cn_hacker.restore()
                self.cfg_hacker.restore()
                self.lora_hacker.restore()
                motion_module.restore(p.sd_model)
                self.hacked = False
                AnimateDiffOutput().output(p, res, params)
                ep_logger.info("AnimateDiff process end.")


def reload_sd_model_vae(sd_model, vae):
    """Reload sd model and vae"""
    shared.opts.sd_model_checkpoint = sd_model
    sd_models.reload_model_weights()
    shared.opts.sd_vae = vae
    sd_vae.reload_vae_weights()


def refresh_model_vae():
    """Refresh sd model and vae"""
    list_models()
    refresh_vae_list()
    try:
        from scripts import global_state

        global_state.update_cn_models()
    except Exception as e:
        ep_logger.error(f"Controlnet Refresh Error. Error info:{e}")
        traceback.print_exc()


@switch_return_opt()
def t2i_call(
    resize_mode=0,
    prompt="",
    styles=[],
    seed=-1,
    subseed=-1,
    subseed_strength=0,
    seed_resize_from_h=0,
    seed_resize_from_w=0,
    batch_size=1,
    n_iter=1,
    steps=None,
    cfg_scale=7.0,
    width=640,
    height=768,
    restore_faces=False,
    tiling=False,
    do_not_save_samples=False,
    do_not_save_grid=False,
    negative_prompt="",
    eta=1.0,
    s_churn=0,
    s_tmax=0,
    s_tmin=0,
    s_noise=1,
    override_settings={},
    override_settings_restore_afterwards=True,
    sampler=None,  # deprecated: use sampler_name
    include_init_images=False,
    controlnet_units: List[ControlNetUnit] = [],
    use_deprecated_controlnet=False,
    outpath_samples="",
    sd_vae=None,
    sd_model_checkpoint="Chilloutmix-Ni-pruned-fp16-fix.safetensors",
    animatediff_flag=False,
    animatediff_video_length=0,
    animatediff_fps=0,
    loractl_flag=False,
):
    """
    Perform text-to-image generation.

    Args:
        resize_mode (int): Resize mode.
        prompt (str): Prompt text.
        styles (list): List of styles.
        seed (int): Seed value.
        subseed (int): Subseed value.
        subseed_strength (int): Subseed strength.
        seed_resize_from_h (int): Seed resize height.
        seed_resize_from_w (int): Seed resize width.
        batch_size (int): Batch size.
        n_iter (int): Number of iterations.
        steps (list): List of steps.
        cfg_scale (float): Configuration scale.
        width (int): Output image width.
        height (int): Output image height.
        restore_faces (bool): Restore faces flag.
        tiling (bool): Tiling flag.
        do_not_save_samples (bool): Do not save samples flag.
        do_not_save_grid (bool): Do not save grid flag.
        negative_prompt (str): Negative prompt text.
        eta (float): Eta value.
        s_churn (int): Churn value.
        s_tmax (int): Tmax value.
        s_tmin (int): Tmin value.
        s_noise (int): Noise value.
        override_settings (dict): Dictionary of override settings.
        override_settings_restore_afterwards (bool): Flag to restore override settings afterwards.
        sampler (object): Sampler object.
        include_init_images (bool): Include initial images flag.
        controlnet_units (List[ControlNetUnit]): List of control net units.
        use_deprecated_controlnet (bool): Use deprecated control net flag.
        outpath_samples (str): Output path for samples.
        sd_vae (str): VAE model checkpoint.
        sd_model_checkpoint (str): Model checkpoint.
        animatediff_flag (bool): Animatediff flag.
        animatediff_video_length (int): Animatediff video length.
        animatediff_fps (int): Animatediff video FPS.
        loractl_flag (bool): Whether to append LoRA weight in all steps to `gen_image` or not.

    Returns:
        gen_image (Union[PIL.Image.Image, List[PIL.Image.Image]]): Generated image.
    """
    if sampler is None:
        sampler = "Euler a"
    if steps is None:
        steps = 20

    # Pass sd_model to StableDiffusionProcessingTxt2Img does not work.
    # We should modify shared.opts.sd_model_checkpoint instead.
    p_txt2img = StableDiffusionProcessingTxt2Img(
        outpath_samples=outpath_samples,
        do_not_save_samples=do_not_save_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=prompt,
        negative_prompt=negative_prompt,
        styles=[],
        seed=seed,
        subseed=subseed,
        subseed_strength=subseed_strength,
        seed_resize_from_h=seed_resize_from_h,
        seed_resize_from_w=seed_resize_from_w,
        sampler_name=sampler,
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        restore_faces=restore_faces,
        tiling=tiling,
        override_settings=override_settings,
    )

    p_txt2img.scripts = scripts.scripts_txt2img
    p_txt2img.script_args = init_default_script_args(p_txt2img.scripts)

    shared.opts.return_mask = False
    shared.opts.return_mask_composite = False
    if "control_net_no_detectmap" in shared.opts.data.keys():
        shared.opts.data["control_net_no_detectmap"] = True

    if animatediff_flag:
        before_pad_cond_uncond = copy.deepcopy(opts.pad_cond_uncond)
        opts.pad_cond_uncond = True

        animate_diff_process = AnimateDiffProcess(enable=True, video_length=animatediff_video_length, fps=animatediff_fps)
        controlnet_units = [ControlNetUnit(**controlnet_unit) for controlnet_unit in controlnet_units]
    else:
        animate_diff_process = None

    for alwayson_scripts in modules.scripts.scripts_img2img.alwayson_scripts:
        if hasattr(alwayson_scripts, "name"):
            if alwayson_scripts.name is None:
                continue
            if alwayson_scripts.name == "controlnet":
                p_txt2img.script_args[alwayson_scripts.args_from : alwayson_scripts.args_from + len(controlnet_units)] = controlnet_units
            if alwayson_scripts.name == "animatediff_easyphoto" and animate_diff_process is not None:
                p_txt2img.script_args[alwayson_scripts.args_from] = animate_diff_process
            if "dynamic lora weights" in alwayson_scripts.name:
                # All LoRAs in the additional prompt will be wrapped with LoraCtlNetwork rather than ExtraNetworkLora.
                p_txt2img.script_args[alwayson_scripts.args_from] = True  # Enable Dynamic Lora Weights
                p_txt2img.script_args[alwayson_scripts.args_from + 1] = loractl_flag  # Plot the LoRA weight in all steps
        else:
            if alwayson_scripts.title().lower() is None:
                continue
            if alwayson_scripts.title().lower() == "controlnet":
                p_txt2img.script_args[alwayson_scripts.args_from : alwayson_scripts.args_from + len(controlnet_units)] = controlnet_units
            if alwayson_scripts.title().lower() == "animatediff_easyphoto" and animate_diff_process is not None:
                p_txt2img.script_args[alwayson_scripts.args_from] = animate_diff_process
            if "dynamic lora weights" in alwayson_scripts.title().lower():
                # All LoRAs in the additional prompt will be wrapped with LoraCtlNetwork rather than ExtraNetworkLora.
                p_txt2img.script_args[alwayson_scripts.args_from] = True  # Enable Dynamic Lora Weights
                p_txt2img.script_args[alwayson_scripts.args_from + 1] = loractl_flag  # Plot the LoRA weight in all steps
    p_txt2img.seed = int(seed)

    processed = processing.process_images(p_txt2img)

    if animatediff_flag:
        opts.pad_cond_uncond = before_pad_cond_uncond

    return processed.images


@switch_return_opt()
def i2i_inpaint_call(
    images=[],
    resize_mode=0,
    denoising_strength=0.75,
    image_cfg_scale=1.5,
    mask_image=None,  # PIL Image mask
    mask_blur=8,
    inpainting_fill=0,
    inpaint_full_res=True,
    inpaint_full_res_padding=0,
    inpainting_mask_invert=0,
    initial_noise_multiplier=1,
    prompt="",
    styles=[],
    seed=-1,
    subseed=-1,
    subseed_strength=0,
    seed_resize_from_h=0,
    seed_resize_from_w=0,
    batch_size=1,
    n_iter=1,
    steps=None,
    cfg_scale=7.0,
    width=640,
    height=768,
    restore_faces=False,
    tiling=False,
    do_not_save_samples=False,
    do_not_save_grid=False,
    negative_prompt="",
    eta=1.0,
    s_churn=0,
    s_tmax=0,
    s_tmin=0,
    s_noise=1,
    override_settings={},
    override_settings_restore_afterwards=True,
    sampler=None,
    include_init_images=False,
    controlnet_units: List[ControlNetUnit] = [],
    use_deprecated_controlnet=False,
    outpath_samples="",
    sd_vae="vae-ft-mse-840000-ema-pruned.ckpt",
    sd_model_checkpoint="Chilloutmix-Ni-pruned-fp16-fix.safetensors",
    animatediff_flag=False,
    animatediff_video_length=0,
    animatediff_fps=0,
    animatediff_reserve_scale=1,
    animatediff_last_image=None,
    loractl_flag=False,
):
    """
    Perform image-to-image inpainting.

    Args:
        images (list): List of input images.
        resize_mode (int): Resize mode.
        denoising_strength (float): Denoising strength.
        image_cfg_scale (float): Image configuration scale.
        mask_image (PIL.Image.Image): Mask image.
        mask_blur (int): Mask blur strength.
        inpainting_fill (int): Inpainting fill value.
        inpaint_full_res (bool): Flag to inpaint at full resolution.
        inpaint_full_res_padding (int): Padding size for full resolution inpainting.
        inpainting_mask_invert (int): Invert the mask flag.
        initial_noise_multiplier (int): Initial noise multiplier.
        prompt (str): Prompt text.
        styles (list): List of styles.
        seed (int): Seed value.
        subseed (int): Subseed value.
        subseed_strength (int): Subseed strength.
        seed_resize_from_h (int): Seed resize height.
        seed_resize_from_w (int): Seed resize width.
        batch_size (int): Batch size.
        n_iter (int): Number of iterations.
        steps (list): List of steps.
        cfg_scale (float): Configuration scale.
        width (int): Output image width.
        height (int): Output image height.
        restore_faces (bool): Restore faces flag.
        tiling (bool): Tiling flag.
        do_not_save_samples (bool): Do not save samples flag.
        do_not_save_grid (bool): Do not save grid flag.
        negative_prompt (str): Negative prompt text.
        eta (float): Eta value.
        s_churn (int): Churn value.
        s_tmax (int): Tmax value.
        s_tmin (int): Tmin value.
        s_noise (int): Noise value.
        override_settings (dict): Dictionary of override settings.
        override_settings_restore_afterwards (bool): Flag to restore override settings afterwards.
        sampler: Sampler.
        include_init_images (bool): Include initial images flag.
        controlnet_units (List[ControlNetUnit]): List of control net units.
        use_deprecated_controlnet (bool): Use deprecated control net flag.
        outpath_samples (str): Output path for samples.
        sd_vae (str): VAE model checkpoint.
        sd_model_checkpoint (str): Model checkpoint.
        animatediff_flag (bool): Animatediff flag.
        animatediff_video_length (int): Animatediff video length.
        animatediff_fps (int): Animatediff video FPS.
        loractl_flag (bool): Whether to append LoRA weight in all steps to `gen_image` or not.

    Returns:
        gen_image (Union[PIL.Image.Image, List[PIL.Image.Image]]): Generated image.
    """
    if sampler is None:
        sampler = "Euler a"
    if steps is None:
        steps = 20

    # Pass sd_model to StableDiffusionProcessingTxt2Img does not work.
    # We should modify shared.opts.sd_model_checkpoint instead.
    p_img2img = StableDiffusionProcessingImg2Img(
        outpath_samples=outpath_samples,
        do_not_save_samples=do_not_save_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_img2img_grids,
        prompt=prompt,
        negative_prompt=negative_prompt,
        styles=[],
        seed=seed,
        subseed=subseed,
        subseed_strength=subseed_strength,
        seed_resize_from_h=seed_resize_from_h,
        seed_resize_from_w=seed_resize_from_w,
        sampler_name=sampler,
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        restore_faces=restore_faces,
        tiling=tiling,
        init_images=images,
        mask=mask_image,
        mask_blur=mask_blur,
        inpainting_fill=inpainting_fill,
        resize_mode=resize_mode,
        denoising_strength=denoising_strength,
        image_cfg_scale=image_cfg_scale,
        inpaint_full_res=inpaint_full_res,
        inpaint_full_res_padding=inpaint_full_res_padding,
        inpainting_mask_invert=inpainting_mask_invert,
        override_settings=override_settings,
        initial_noise_multiplier=initial_noise_multiplier,
    )

    p_img2img.scripts = scripts.scripts_img2img
    p_img2img.extra_generation_params["Mask blur"] = mask_blur
    p_img2img.script_args = init_default_script_args(p_img2img.scripts)

    shared.opts.return_mask = False
    shared.opts.return_mask_composite = False
    if "control_net_no_detectmap" in shared.opts.data.keys():
        shared.opts.data["control_net_no_detectmap"] = True

    if animatediff_flag:
        before_pad_cond_uncond = copy.deepcopy(opts.pad_cond_uncond)
        opts.pad_cond_uncond = True

        animate_diff_process = AnimateDiffProcess(
            enable=True,
            video_length=len(images) if animatediff_video_length == 0 else animatediff_video_length,
            fps=animatediff_fps,
            i2i_reserve_scale=animatediff_reserve_scale,
            last_frame=animatediff_last_image,
            latent_scale=len(images) if animatediff_video_length == 0 else int(animatediff_video_length / 4 * 3),
            latent_scale_last=len(images) if animatediff_video_length == 0 else int(animatediff_video_length / 4 * 1),
        )
        controlnet_units = [ControlNetUnit(**controlnet_unit) for controlnet_unit in controlnet_units]
    else:
        animate_diff_process = None

    for alwayson_scripts in modules.scripts.scripts_img2img.alwayson_scripts:
        if hasattr(alwayson_scripts, "name"):
            if alwayson_scripts.name is None:
                continue
            if alwayson_scripts.name == "controlnet":
                p_img2img.script_args[alwayson_scripts.args_from : alwayson_scripts.args_from + len(controlnet_units)] = controlnet_units
            if alwayson_scripts.name == "animatediff_easyphoto" and animate_diff_process is not None:
                p_img2img.script_args[alwayson_scripts.args_from] = animate_diff_process
            if "dynamic lora weights" in alwayson_scripts.name:
                # All LoRAs in the additional prompt will be wrapped with LoraCtlNetwork rather than ExtraNetworkLora.
                p_img2img.script_args[alwayson_scripts.args_from] = True  # Enable Dynamic Lora Weights
                p_img2img.script_args[alwayson_scripts.args_from + 1] = loractl_flag  # Plot the LoRA weight in all steps
        else:
            if alwayson_scripts.title().lower() is None:
                continue
            if alwayson_scripts.title().lower() == "controlnet":
                p_img2img.script_args[alwayson_scripts.args_from : alwayson_scripts.args_from + len(controlnet_units)] = controlnet_units
            if alwayson_scripts.title().lower() == "animatediff_easyphoto" and animate_diff_process is not None:
                p_img2img.script_args[alwayson_scripts.args_from] = animate_diff_process
            if "dynamic lora weights" in alwayson_scripts.title().lower():
                # All LoRAs in the additional prompt will be wrapped with LoraCtlNetwork rather than ExtraNetworkLora.
                p_img2img.script_args[alwayson_scripts.args_from] = True  # Enable Dynamic Lora Weights
                p_img2img.script_args[alwayson_scripts.args_from + 1] = loractl_flag  # Plot the LoRA weight in all steps
    p_img2img.seed = int(seed)

    processed = processing.process_images(p_img2img)

    if animatediff_flag:
        opts.pad_cond_uncond = before_pad_cond_uncond

    return processed.images


def get_checkpoint_type(sd_model_checkpoint: str) -> int:
    """Get the type of the stable diffusion model given the checkpoint name.

    Args:
        sd_model_checkpoint (str): the checkpoint name.

    Returns:
        An integer representing the model type (1 means SD1; 2 means SD2; 3 means SDXL).
    """
    ckpt_path = os.path.join(models_path, "Stable-diffusion", sd_model_checkpoint)
    timer = Timer()
    checkpoint_info = sd_models.CheckpointInfo(ckpt_path)
    state_dict = sd_models.get_checkpoint_state_dict(checkpoint_info, timer)
    for k in state_dict.keys():
        # SD web UI uses `hasattr(model, conditioner)` to check the SDXL model.
        if k.startswith("conditioner"):
            return 3
        # SD web UI uses `hasattr(model, model.cond_stage_model)` to check the SD2 model.
        if k.startswith("cond_stage_model.model"):
            return 2
    return 1


def read_lora_metadata(filename: str) -> str:
    """Read the metadata from the Lora given the path `filename`. Modified from `extensions-builtin/Lora/network.py`.

    Args:
        filename (str): the Lora file path.

    Returns:
        A string representing the metadata.
    """
    # Firstly, read the metadata of the Lora from the cache. If the Lora is added to the folder
    # after the SD Web UI launches, then read the Lora from the hard disk to get the metadata.
    try:
        name = os.path.splitext(os.path.basename(filename))[0]
        read_metadata = lambda filename: sd_models.read_metadata_from_safetensors(filename)
        # It will return None if the Lora file has not be cached before.
        metadata = cache.cached_data_for_file("safetensors-metadata", "lora/" + name, filename, read_metadata)

        return metadata
    except TypeError:
        metadata = sd_models.read_metadata_from_safetensors(filename)

        return metadata
    except Exception as e:
        errors.display(e, f"reading lora {filename}")


def get_lora_type(filename: str) -> int:
    """Get the type of the Lora given the path `filename`.

    Args:
        filename (str): the Lora file path.

    Returns:
        An integer representing the Lora type (1 means SD1; 2 means SD2; 3 means SDXL).
    """
    metadata = read_lora_metadata(filename)
    if str(metadata.get("ss_base_model_version", "")).startswith("sdxl_"):
        return 3
    elif str(metadata.get("ss_v2", "")) == "True":
        return 2
    return 1


def get_scene_prompt(filename: str) -> str:
    """Check the scene lora and get the scene prompt from the path `filename`.

    Args:
        filename (str): the Lora file path.

    Returns:
        A string represents the scene prompt.
    """
    metadata = read_lora_metadata(filename)
    if str(metadata.get("ep_lora_version", "")).startswith("scene"):
        prompt = str(metadata.get("ep_prompt", ""))
        # Convert a list of strings (type: str) into an list of strings (type: List).
        if prompt[0] == "[" and prompt[-1] == "]":
            prompt_list = re.findall(r"'(.*?)'", prompt)
            prompt = random.choice(prompt_list)
        return prompt
    return ""
