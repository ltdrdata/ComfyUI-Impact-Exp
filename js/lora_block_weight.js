import { ComfyApp, app } from "../../scripts/app.js";

app.registerExtension({
	name: "Comfy.Impact.Exp",

	nodeCreated(node, app) {
		if(node.comfyClass == "ImpactLoraLoaderBlockWeight") {
			node._value = "Preset";
			Object.defineProperty(node.widgets[4], "value", {
				set: (value) => {
				        const stackTrace = new Error().stack;
                        if(stackTrace.includes('inner_value_change')) {
                            if(value != "Preset") {
                                node.widgets[5].value = value.split(':')[1];
	                            if(node.widgets_values) {
	                                node.widgets_values[5] = node.widgets[4].value;
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

		if(node.comfyClass == "ImpactXYPlotLoraBlockWeight") {
			node._value = "Preset";
			Object.defineProperty(node.widgets[4], "value", {
				set: (value) => {
				        const stackTrace = new Error().stack;
                        if(stackTrace.includes('inner_value_change')) {
                            if(value != "Preset") {
                                if(node.widgets[5].value != "")
                                    node.widgets[5].value += "\n";
                                node.widgets[5].value += `${value}/${value.split(':')[0]}`;
	                            if(node.widgets_values) {
	                                node.widgets_values[5] = node.widgets[4].value;
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