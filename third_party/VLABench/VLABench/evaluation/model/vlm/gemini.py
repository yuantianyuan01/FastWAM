from VLABench.evaluation.model.vlm.base import *

class Gemini(BaseVLM):
    def __init__(self, api_key=None, base_url=None, model="gemini-2.5-pro-exp-03-25") -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        super().__init__()

    def evaluate(self, input_dict, language, with_CoT=False):
        from VLABench.utils.gpt_utils import build_prompt_with_tilist, query_gpt4_v
        ti_list = []
        ti_list.append(["text", input_dict["pre_prompt"] ])

        ti_list = get_ti_list(input_dict, language, with_CoT)

        prompt = build_prompt_with_tilist(ti_list)
        content = query_gpt4_v(prompt, api_key=self.api_key, base_url=self.base_url, model=self.model)
        output = {}
        output["origin_output"] = content
        try:
            json_data = content.split("```json")[1].split("```")[0]
            output["skill_sequence"] = json.loads(json_data)
        except:
            output["format_error"] = "format_error"
        return output

    def get_name(self):
        return "Gemini"