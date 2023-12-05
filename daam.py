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