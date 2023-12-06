from torch import nn

class AttentionLogger:
    def __init__(self, heat_maps):
        self.heat_maps = heat_maps

    def __call__(self, q, k, v, extra_options):
        org_dtype = q.dtype

        with torch.autocast(device_type=self.device, dtype=self.dtype):
            out = optimized_attention(q, k, v, extra_options["n_heads"])
            _, _, lh, lw = extra_options["original_shape"]
            
            attention_probs = attn.get_attention_scores(query, key, attention_mask)

            if attention_probs.shape[-1] == self.context_size and factor != 8:
                pass
                # # shape: (batch_size, 64 // factor, 64 // factor, 77)
                # maps = self._unravel_attn(attention_probs)

                # for head_idx, heatmap in enumerate(maps):
                #     self.heat_maps.update(factor, self.layer_idx, head_idx, heatmap)

            return out.to(dtype=org_dtype)


def set_model_patch_replace(model, patch_kwargs, key):
    to = model.model_options["transformer_options"]
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    if "attn2" not in to["patches_replace"]:
        to["patches_replace"]["attn2"] = {}
    if key not in to["patches_replace"]["attn2"]:
        patch = AttentionLogger(**patch_kwargs)
        to["patches_replace"]["attn2"][key] = patch
    else:
        to["patches_replace"]["attn2"][key].set_new_condition(**patch_kwargs)


class DaamTest:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL", ),
                     "text": ("STRING", {"multiline": True}), "clip": ("CLIP", )}
                }

    RETURN_TYPES = ("STRING", )
    FUNCTION = "doit"

    CATEGORY = "Exp"

    def doit(self, model, text):
        return (model, )


NODE_CLASS_MAPPINGS = {
    "DaamTest": DaamTest,
}
NODE_DISPLAY_NAME_MAPPINGS = {
}