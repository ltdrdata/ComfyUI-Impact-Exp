import inspect

class FlexTest:
    @classmethod
    def INPUT_TYPES(s):
        flex_inputs = {}

        stack = inspect.stack()

        if stack[1].function == 'get_input_data':
            # bypass validation
            for x in range(0, 20):
                flex_inputs[f"input{x}"] = (["aaa", "bbb", "ccc"],)

        return {"required":
                    { "count": ("INT", {"default": 0, "min": 0, "max": 10}) },
                "optional":
                    flex_inputs
                }

    RETURN_TYPES = ("STRING", )
    FUNCTION = "doit"

    CATEGORY = "Exp"

    def doit(self, **kwargs):
        temp = ""
        for k, v in kwargs.items():
            temp += f"/ {k}, {v}"

        return (temp, )


NODE_CLASS_MAPPINGS = {
    "FlexTest": FlexTest,
}
NODE_DISPLAY_NAME_MAPPINGS = {
}