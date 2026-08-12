'''
Pass the local image base64 code to the OpenAI API to get the image captioning result.
'''
from openai import OpenAI
import os
import base64
import json

def convert_base64_to_data_uri(base64_image):
    def _get_mime_type_from_data_uri(base64_image):
        # Decode the base64 string
        image_data = base64.b64decode(base64_image)
        # Check the first few bytes for known signatures
        if image_data.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        elif image_data.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        elif image_data.startswith(b"GIF87a") or image_data.startswith(b"GIF89a"):
            return "image/gif"
        elif image_data.startswith(b"RIFF") and image_data[8:12] == b"WEBP":
            return "image/webp"
        return "image/jpeg"  # use jpeg for unknown formats, best guess.

    mime_type = _get_mime_type_from_data_uri(base64_image)
    data_uri = f"data:{mime_type};base64,{base64_image}"
    return data_uri

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def query_gpt4_v(prompt, history=[], model="gpt-4-turbo", **kwargs):
    client = OpenAI(
        api_key=kwargs.get("api_key", os.environ.get("OPENAI_API_KEY", None)), 
        base_url=kwargs.get("base_url", os.environ.get("OPENAI_BASE_URL", None))
    )
    
    while True:
        try:
            messages = []
            for q,a in history:
                messages.append({"role": "user", "content": q})
                messages.append({"role": "assistant", "content": a})
            messages.append({"role": "user", "content": prompt})
            response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=1000,
                        )
            break
        except Exception as e:
            print(f"openai error, {e}")
    message = response.choices[0].message
    content = message.content
    return content

def build_prompts(text, image_paths):
    prompt = list()
    for image_path in image_paths:
        base64image = encode_image(image_path)
        uri = convert_base64_to_data_uri(base64image)
        prompt.append({"type": "image_url", "image_url": {"url": uri}})
    prompt.append({"type": "text", "text": text})
    return prompt

def build_prompt_with_tilist(text_image_list):
    prompt = list()
    for ti_context in text_image_list:
        if ti_context[0] == 'text':
            text = ti_context[1]
            prompt.append({"type": "text", "text": text})
        elif ti_context[0] == 'image':
            image_path = ti_context[1]
            base64image = encode_image(image_path)
            uri = convert_base64_to_data_uri(base64image)
            prompt.append({"type": "image_url", "image_url": {"url": uri}})
    return prompt