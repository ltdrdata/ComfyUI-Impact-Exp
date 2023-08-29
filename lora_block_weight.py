import folder_paths
import comfy.utils
import comfy.lora


class LoraLoaderBlockWeight:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "clip": ("CLIP", ),
                              "lora_name": (folder_paths.get_filename_list("loras"), ),
                              "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                              "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                              "preset": (
                                  ["sd-lora:1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1"],  # 17
                                  ["sd-etc:1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1"],  # 26
                                  ["sdxl-lora:1,0,0,0,0,0,1,1,1,1,1,1"],  # 12
                                  ["sdxl-etc:1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1"],  # 19
                                ),
                              "block_vector": ("STRING", {"multiline": False, "default": "sd-lora:1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1"}),
                              }}

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "doit"

    CATEGORY = "loaders"

    @staticmethod
    def load_lora_for_models(model, clip, lora, strength_model, strength_clip, block_vector):
        key_map = comfy.lora.model_lora_keys_unet(model.model)
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)
        loaded = comfy.lora.load_lora(lora, key_map)

        block_vector = block_vector.split(":")
        if len(block_vector) > 1:
            block_vector = block_vector[1]

        vector = block_vector.split(",")
        vector_i = 0

        new_modelpatcher = model.clone()
        for k, v in loaded.items():
            ratio = strength_model
            k_unet = k[len("diffusion_model."):]

            k_unet_num = None
            if k_unet.startswith("input_blocks."):
                k_unet_num = k_unet[len("input_blocks."):len("input_blocks.")+2]
            elif k_unet.startswith("middle_block."):
                k_unet_num = k_unet[len("middle_block."):len("middle_block.")+2]
            elif k_unet.startswith("output_blocks."):
                k_unet_num = k_unet[len("output_blocks."):len("output_blocks.")+2]

            if k_unet_num is not None:
                if k_unet_num[1] == '.':
                    k_unet_num = int(k_unet_num[0])
                else:
                    k_unet_num = int(k_unet_num)

                if len(vector) > vector_i:
                    ratio = float(vector[vector_i].strip())

            print(f"\t{k_unet} -> ({ratio}) ")

            new_modelpatcher.add_patches({k: v}, ratio)

        new_modelpatcher.add_patches(loaded, strength_model)
        new_clip = clip.clone()
        new_clip.add_patches(loaded, strength_clip)

        return (new_modelpatcher, new_clip)


    def doit(self, model, clip, lora_name, strength_model, strength_clip, block_vector):
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

        model_lora, clip_lora = LoraLoaderBlockWeight.load_lora_for_models(model, clip, lora, strength_model, strength_clip, block_vector)
        return (model_lora, clip_lora)


NODE_CLASS_MAPPINGS = {
    "LoraLoaderBlockWeight": LoraLoaderBlockWeight
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraLoaderBlockWeight": "LoraLoaderBlockWeight"
}