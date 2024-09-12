import gradio as gr

from modules import scripts, shared

from modules.ui_components import InputAccordion
from modules import prompt_parser, sd_samplers
from modules.processing import StableDiffusionProcessingTxt2Img

class HRCFG_Forge(scripts.Script):
    sorting_priority = 1
    original_CFG = 1.0
    hr_dCFG = 3.5
    hr_dCFG_previous = None
    original_calculate_hr_conds = StableDiffusionProcessingTxt2Img.calculate_hr_conds

    def patched_calculate_hr_conds(self):
        StableDiffusionProcessingTxt2Img.calculate_hr_conds = HRCFG_Forge.original_calculate_hr_conds

        if self.hr_c is not None:
            return

        sampler_config = sd_samplers.find_sampler_config(self.hr_sampler_name or self.sampler_name)
        steps = self.hr_second_pass_steps or self.steps
        total_steps = sampler_config.total_steps(steps) if sampler_config else steps

        if self.cfg_scale == 1:
            self.hr_uc = None
            print('Skipping unconditional conditioning (HR pass) when CFG = 1. Negative Prompts are ignored.')
        else:
            hr_negative_prompts = prompt_parser.SdConditioning(self.hr_negative_prompts, width=self.hr_upscale_to_x, height=self.hr_upscale_to_y, is_negative_prompt=True, distilled_cfg_scale=HRCFG_Forge.hr_dCFG)
            self.hr_uc = self.get_conds_with_caching(prompt_parser.get_learned_conditioning, hr_negative_prompts, self.firstpass_steps, [self.cached_hr_uc, self.cached_uc], self.hr_extra_network_data, total_steps)

        hr_prompts = prompt_parser.SdConditioning(self.hr_prompts, width=self.hr_upscale_to_x, height=self.hr_upscale_to_y, distilled_cfg_scale=HRCFG_Forge.hr_dCFG)
        self.hr_c = self.get_conds_with_caching(prompt_parser.get_multicond_learned_conditioning, hr_prompts, self.firstpass_steps, [self.cached_hr_c, self.cached_c], self.hr_extra_network_data, total_steps)

    def title(self):
        return "HighRes fix: override CFG and Distilled CFG"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with InputAccordion(False, label=self.title()) as enabled:
            hrcfg = gr.Slider(label='CFG for HighRes pass', minimum=1.0, maximum=30.0, step=0.1, value=1.0)
            hrdcfg = gr.Slider(label='Distilled CFG for HighRes pass', minimum=0.0, maximum=30.0, step=0.1, value=1.0)

        self.infotext_fields = [
            (enabled, lambda d: d.get("HRCFG_enabled", False)),
            (hrcfg,   "HiresCFG"),
            (hrdcfg,  "HiresDCFG"),
        ]

        return enabled, hrcfg, hrdcfg

    def before_process(self, p, *args):
        enabled, _, HRCFG_Forge.hr_dCFG = args

        if enabled:
            if HRCFG_Forge.hr_dCFG != HRCFG_Forge.hr_dCFG_previous:
                p.cached_hr_c = [None, None, None]
                p.cached_hr_uc = [None, None, None]

            StableDiffusionProcessingTxt2Img.calculate_hr_conds = HRCFG_Forge.patched_calculate_hr_conds

        return

    def process(self, p, *args, **kwargs):
        enabled, hr_CFG, _ = args

        if enabled:
            p.extra_generation_params.update(dict(
                HRCFG_enabled = enabled,
                HiresCFG      = hr_CFG,
            ))

            if not shared.sd_model.is_webui_legacy_model():
                p.extra_generation_params.update(dict(
                    HiresDCFG      = HRCFG_Forge.hr_dCFG,
                ))

        return

    def before_hr(self, p, *args):
        enabled, hr_CFG, _ = args

        if enabled:
            HRCFG_Forge.original_CFG = p.cfg_scale
            p.cfg_scale = hr_CFG

        return

    def postprocess_image(self, p, pp, *args):
        enabled = args[0]

        if enabled:
            p.cfg_scale = HRCFG_Forge.original_CFG
            HRCFG_Forge.hr_dCFG_previous = HRCFG_Forge.hr_dCFG

        return
