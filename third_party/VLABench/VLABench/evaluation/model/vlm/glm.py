from VLABench.evaluation.model.vlm.base import *

class GLM4v(BaseVLM):
    def __init__(self) -> None:
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        MODEL_PATH = "THUDM/glm-4v-9b"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto"
        ).eval()

    def evaluate(self, input_dict, language, with_CoT=False):
        ti_list = get_ti_list(input_dict, language, with_CoT)
        content, combined_image = self.build_prompt_with_tilist(ti_list)
        inputs = self.tokenizer.apply_chat_template([{"role": "user", "image": combined_image, "content": content}],
                                            add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                            return_dict=True)  # chat mode

        inputs = inputs.to(self.device)

        gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            output_text = self.tokenizer.decode(outputs[0])
            output = {}
            output["origin_output"] = output_text
            try:
                json_data = output_text.split("```json")[1].split("```")[0]
                # print(json_data)
                output["skill_sequence"] = json.loads(json_data)
            except:
                # print("No json data found")
                output["format_error"] = "format_error"
            return output

    
    def build_prompt_with_tilist(self, ti_list):
        from PIL import Image
        content = ""
        image_list = []
        for ti in ti_list:
            if ti[0] == "text":
                content += ti[1] + "\n"
            elif ti[0] == "image":
                image_list.append(ti[1])
            
        images = [Image.open(image_path).convert('RGB') for image_path in image_list]
        width, height = images[0].size
        combined_image = Image.new('RGB', (width * len(images), height))

        for i, img in enumerate(images):
            combined_image.paste(img, (i * width, 0))

        return content, combined_image

    def get_name(self):
        return "GLM4v"