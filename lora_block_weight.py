import folder_paths
import comfy.utils
import comfy.lora
import os
import torch
import numpy as np
import nodes
from PIL import Image


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def load_lbw_preset(filename):
    path = os.path.join(os.path.dirname(__file__), filename)
    preset_list = []

    if os.path.exists(path):
        with open(path, 'r') as file:
            for line in file:
                preset_list.append(line.strip())

        return preset_list
    else:
        return []


class LoraLoaderBlockWeight:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        preset = ["Preset"]  # 20
        preset += load_lbw_preset("lbw-preset.txt")
        preset += load_lbw_preset("lbw-preset.custom.txt")

        return {"required": { "model": ("MODEL",),
                              "clip": ("CLIP", ),
                              "lora_name": (folder_paths.get_filename_list("loras"), ),
                              "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                              "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                              "inverse": ("BOOLEAN", {"default": False, "label_on": "True", "label_off": "False"}),
                              "preset": (preset,),
                              "block_vector": ("STRING", {"multiline": True, "default": "1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1"}),
                              }}

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "doit"

    CATEGORY = "__impact_exp"

    @staticmethod
    def load_lora_for_models(model, clip, lora, strength_model, strength_clip, inverse, block_vector):
        key_map = comfy.lora.model_lora_keys_unet(model.model)
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)
        loaded = comfy.lora.load_lora(lora, key_map)

        block_vector = block_vector.split(":")
        if len(block_vector) > 1:
            block_vector = block_vector[1]
        else:
            block_vector = block_vector[0]

        vector = block_vector.split(",")
        vector_i = 1

        last_k_unet_num = None
        new_modelpatcher = model.clone()
        ratio = strength_model

        def parse_unet_num(s):
            if s[1] == '.':
                return int(s[0])
            else:
                return int(s)

        # sort: input, middle, output, others
        input_blocks = []
        middle_blocks = []
        output_blocks = []
        others = []
        for k, v in loaded.items():
            k_unet = k[len("diffusion_model."):]

            if k_unet.startswith("input_blocks."):
                k_unet_num = k_unet[len("input_blocks."):len("input_blocks.")+2]
                input_blocks.append((k, v, parse_unet_num(k_unet_num), k_unet))
            elif k_unet.startswith("middle_block."):
                k_unet_num = k_unet[len("middle_block."):len("middle_block.")+2]
                middle_blocks.append((k, v, parse_unet_num(k_unet_num), k_unet))
            elif k_unet.startswith("output_blocks."):
                k_unet_num = k_unet[len("output_blocks."):len("output_blocks.")+2]
                output_blocks.append((k, v, parse_unet_num(k_unet_num), k_unet))
            else:
                others.append((k, v, k_unet))

        input_blocks = sorted(input_blocks, key=lambda x: x[2])
        middle_blocks = sorted(middle_blocks, key=lambda x: x[2])
        output_blocks = sorted(output_blocks, key=lambda x: x[2])

        # prepare patch
        for k, v, k_unet_num, k_unet in (input_blocks + middle_blocks + output_blocks):
            if last_k_unet_num != k_unet_num and len(vector) > vector_i:
                ratio = float(vector[vector_i].strip())
                vector_i += 1

            last_k_unet_num = k_unet_num

            if inverse:
                new_modelpatcher.add_patches({k: v}, strength_model * (1 - ratio))
                print(f"\t{k_unet} -> inv({ratio}) ")
            else:
                new_modelpatcher.add_patches({k: v}, strength_model * ratio)
                print(f"\t{k_unet} -> ({ratio}) ")

        # prepare base patch
        ratio = float(vector[0].strip())
        for k, v, k_unet in others:

            if inverse:
                new_modelpatcher.add_patches({k: v}, strength_model * (1 - ratio))
                print(f"\t{k_unet} -> inv({ratio}) ")
            else:
                new_modelpatcher.add_patches({k: v}, strength_model * ratio)
                print(f"\t{k_unet} -> ({ratio}) ")

        new_clip = clip.clone()
        new_clip.add_patches(loaded, strength_clip)

        return (new_modelpatcher, new_clip)

    def doit(self, model, clip, lora_name, strength_model, strength_clip, inverse, preset, block_vector):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = LoraLoaderBlockWeight.load_lora_for_models(model, clip, lora, strength_model, strength_clip, inverse, block_vector)
        return (model_lora, clip_lora)


class XY_Capsule:
    def pre_define_model(self, model, clip, vae):
        return model, clip, vae

    def get_result(self, model, clip, vae):
        return None

    def set_another_capsule(self, capsule):
        return None

    def getLabel(self):
        return "Unknown"


class XY_Capsule_LoraBlockWeight(XY_Capsule):
    def __init__(self, x, y, label, storage):
        self.x = x
        self.y = y
        self.label = label
        self.storage = []
        self.another_capsule = None

    def set_another_capsule(self, capsule):
        self.another_capsule = capsule

    def patch_model(self, model):
        return model

    def pre_define_model(self, model, clip, vae):
        if self.y == 0:
            return self.patch_model(model)
        elif self.y == 1:
            return self.patch_model(model)
        else:
            return model, clip, vae

    def get_result(self, model, clip, vae):
        if self.y < 2:
            return None

        if self.y == 2:
            # diff
            weighted_image = self.storage[self.x][0]
            reference_image = self.storage[self.x][1]
            image = torch.abs(weighted_image - reference_image)
            self.storage[self.x][self.y] = image
        elif self.y == 3:
            # heatmap
            heatmap = self.storage[self.x][2].squeeze().sum(dim=-1)
            image = self.storage[self.x][0].clone()
            image += 0.5 * heatmap

        latent = nodes.VAEEncode().encode(vae, image)[0]
        return (latent, image)


    def getLabel(self):
        return self.label


class XYPlot_LoraBlockWeight:
    @classmethod
    def INPUT_TYPES(cls):
        preset = ["Preset"]  # 20
        preset += load_lbw_preset("lbw-preset.txt")
        preset += load_lbw_preset("lbw-preset.custom.txt")

        default_vectors = "SD-NONE\nSD-ALL\nSD-INS\nSD-IND\nSD-INALL\nSD-MIDD\nSD-MIDD0.2\nSD-MIDD0.8\nSD-MOUT\nSD-OUTD\nSD-OUTS\nSD-OUTALL"

        return {"required": { "model": ("MODEL",),
                              "clip": ("CLIP", ),
                              "lora_name": (folder_paths.get_filename_list("loras"), ),
                              "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                              "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                              "inverse": ("BOOLEAN", {"default": False, "label_on": "True", "label_off": "False"}),
                              "preset": (preset,),
                              "block_vectors": ("STRING", {"multiline": True, "default": default_vectors}),
                              }}

    RETURN_TYPES = ("XY", "XY"),
    RETURN_NAMES = ("X (vectors)", "Y (effect_compares)")

    FUNCTION = "xy_value"
    CATEGORY = "__impact_exp"

    def doit(self, model, clip, lora_name, strength_model, strength_clip, inverse, preset, block_vectors):
        xy_type = "XY_Capsule"

        xy_value = []

        for block_vectors in preset.split("\n"):
            block_vector = block_vectors.strip()

            if block_vector == "":
                XY_Capsule()
                continue

        return ((xy_type, xy_value), )


NODE_CLASS_MAPPINGS = {
    "LoraLoaderBlockWeight": LoraLoaderBlockWeight
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraLoaderBlockWeight": "LoraLoaderBlockWeight"
}
