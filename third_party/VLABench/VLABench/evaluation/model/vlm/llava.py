from VLABench.evaluation.model.vlm.base import *

class Llava_NeXT(BaseVLM):
    def __init__(self) -> None:
        from transformers import AutoProcessor, LlavaNextForConditionalGeneration
        super().__init__()
        self.model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, device_map="auto"
                                                          ,use_flash_attention_2=True
                                                          )
        self.processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

    def evaluate(self, input_dict, language, with_CoT=False):
        few_shot_num = len(input_dict["shot_input_pic"].keys())
        ti_list = get_ti_list(input_dict, language, with_CoT=with_CoT)
        content, image_list = self.build_prompt_with_tilist(ti_list)
        conversation = [
            {
                "role": "user",
                "content": content,
            }
        ]
        new_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        # We can simply feed images in the order they have to be used in the text prompt
        # Each "<image>" token uses one image leaving the next for the subsequent "<image>" tokens
        inputs = self.processor(images=image_list, text=new_prompt, padding=True, return_tensors="pt").to(self.model.device)

        # Generate
        generate_ids = self.model.generate(**inputs, max_new_tokens=200)
        output_text = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        image_num = few_shot_num*2 +2

        output = {}
        output["origin_output"] = output_text[len(new_prompt)-image_num*7+(image_num):]
        try:
            json_data = output["origin_output"].split("```json")[1].split("```")[0]
            # print(json_data)
            output["skill_sequence"] = json.loads(json_data)
        except:
            # print("No json data found")
            output["format_error"] = "format_error"
        return output
    
    def build_prompt_with_tilist(self, ti_list):
        from PIL import Image
        content = []
        image_list = []
        for ti in ti_list:
            if ti[0] == "text":
                content.append({"type": "text", "text": ti[1]})
            elif ti[0] == "image":
                content.append({"type": "image"})
                image_list.append(Image.open(ti[1]))
        return content, image_list
    
    def get_name(self):
        return "Llava_NeXT"