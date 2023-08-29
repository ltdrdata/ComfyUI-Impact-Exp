import { ComfyApp, app } from "../../scripts/app.js";

app.registerExtension({
	name: "Comfy.Impact.Exp",

	nodeCreated(node, app) {
		if(node.comfyClass == "LoraLoaderBlockWeight") {
			node._value = "Preset";
			Object.defineProperty(node.widgets[3], "value", {
				set: (value) => {
				        const stackTrace = new Error().stack;
                        if(stackTrace.includes('inner_value_change')) {
                            if(value != "Preset") {
                                node.widgets[4].value = value.split(':')[1];
	                            if(node.widgets_values) {
	                                node.widgets_values[4] = node.widgets[3].value;
                                }
                            }
                        }

						node._value = value;
					},
				get: () => {
                        return node._value;
					 }
			});
		}
	}
});