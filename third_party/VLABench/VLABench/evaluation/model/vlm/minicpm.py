from VLABench.evaluation.model.vlm.base import *

class MiniCPM_V2_6(BaseVLM):
    def __init__(self) -> None:
        super().__init__()
        from transformers import AutoTokenizer
        from vllm import LLM
        MODEL_NAME = "openbmb/MiniCPM-V-2_6"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self.llm = LLM(
            model=MODEL_NAME,
            trust_remote_code=True,
            gpu_memory_utilization=1,
            # max_model_len=2048
            max_model_len=4096,
            tensor_parallel_size=2
        )

    def evaluate(self, input_dict, language, with_CoT=False):
        from vllm import SamplingParams
        ti_list = get_ti_list(input_dict, language, with_CoT=with_CoT)
        content, image_list = self.build_prompt_with_tilist(ti_list)

        messages = [{
            "role":"user",
            "content":content
        }]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Single Inference
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {
                "image": image_list
                # Multi images, the number of images should be equal to that of `(<image>./</image>)`
                # "image": [image, image] 
            },
        }

        # 2.6
        stop_tokens = ['<|im_end|>', '<|endoftext|>']
        stop_token_ids = [self.tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]


        sampling_params = SamplingParams(
            stop_token_ids=stop_token_ids, 
            use_beam_search=True,
            temperature=0, 
            best_of=3,
            max_tokens=64
        )

        outputs = self.llm.generate(inputs, sampling_params=sampling_params)

        # print("output")
        # print(outputs[0].outputs[0].text)
        output = {}
        output["origin_output"] = outputs[0].outputs[0].text

        try:
            json_data = outputs[0].outputs[0].text.split("```json")[1].split("```")[0]
            output["skill_sequence"] = json.loads(json_data)
        except:
            try:
                json_data = outputs[0].outputs[0].text
                output["skill_sequence"] = json.loads(json_data)["skill_sequence"]

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
                content+=ti[1] + "\n"
            elif ti[0] == "image":
                content+="(<image>./</image>)" +"\n"
                image_list.append(Image.open(ti[1]).convert("RGB"))
        return content, image_list
    
    def get_name(self):
        return "MiniCPM_V2_6"