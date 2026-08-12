from VLABench.evaluation.model.vlm.base import *

class InternVL2(BaseVLM):
    def __init__(self) -> None:
        super().__init__()
        from lmdeploy import pipeline, TurbomindEngineConfig

        model = 'OpenGVLab/InternVL2-8B'
        self.pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))

    def evaluate(self, input_dict, language, with_CoT=False):
        ti_list = get_ti_list(input_dict, language, with_CoT=with_CoT)
        prompt, images = self.get_prompt_with_tilist(ti_list)
        response = self.pipe((prompt, images))
        # Numbering images improves multi-image conversations
        response = self.pipe((prompt, images))
        # print(response.text)
        output = {}
        output["origin_output"] = response.text
        try:
            json_data = response.text.split("```json")[1].split("```")[0]
            skill_sequence = json.loads(json_data)
            output["skill_sequence"] = skill_sequence
        except:
            output["format_error"] = "format_error"

        return output

    
    def get_prompt_with_tilist(self, ti_list):
        from lmdeploy.vl import load_image
        from lmdeploy.vl.constants import IMAGE_TOKEN
        prompt = ""
        image_urls = []
        for ti in ti_list:
            if ti[0] == "text":
                prompt += ti[1] + "\n"
            elif ti[0] == "image":
                prompt += f'Image: {IMAGE_TOKEN}\n'
                image_urls.append(ti[1])
        images = [load_image(img_url) for img_url in image_urls]
        return prompt, images

    def get_name(self):
        return "InternVL2"