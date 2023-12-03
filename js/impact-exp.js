import { ComfyApp, app } from "../../scripts/app.js";

app.registerExtension({
	name: "Comfy.Impact.Exp",

	nodeCreated(node, app) {
		if(node.comfyClass == "FlexTest") {
			const count_widget = node.widgets.find((w) => w.name == 'count');
			let start_point = 1;

			Object.defineProperty(count_widget, "value", {
                set: async function(value) {
                	let current_widget_count = node.widgets.length - start_point;

                	if(current_widget_count != value) {
						console.log(node.count_value);
						if(current_widget_count > value) {
							let remove_cnt = current_widget_count - value;
							await node.widgets.splice(current_widget_count - remove_cnt + start_point, remove_cnt);
							await node.setSize( node.computeSize() );
						}
						else {
							for(let i=current_widget_count; i<value; i++)
								await node.addWidget("combo", `input${i}`, "aaa", () => {}, { values: ["aaa", "bbb", "ccc"] });
						}
					}
                },
                get: function() {
                	let current_widget_count = node.widgets.length - start_point;
                	return current_widget_count;
                }
			});
		}
	}
});