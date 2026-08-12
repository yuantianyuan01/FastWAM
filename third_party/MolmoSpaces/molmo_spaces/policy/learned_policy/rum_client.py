import io
import json
import logging
import os
import re
import time

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

try:
    import websockets.sync.client as ws_client
except ImportError:
    ws_client = None


class RUMClient:
    def __init__(self, host: str = "localhost", port: int = 8765, max_size: int = 50 * 1024 * 1024):
        if ws_client is None:
            raise ImportError("websockets package required. Install with: pip install websockets")
        self.uri = f"ws://{host}:{port}"
        self.websocket = ws_client.connect(self.uri, max_size=max_size)

    def _send(self, request: dict) -> dict:
        self.websocket.send(json.dumps(request))
        response = json.loads(self.websocket.recv())
        if response.get("status") == "error":
            raise RuntimeError(f"Server error: {response.get('message')}")
        return response

    def reset(self):
        self._send({"action": "reset"})

    def infer(self, observation: dict) -> np.ndarray:
        request = {
            "action": "infer",
            "observation": {
                "rgb_ego": observation["rgb_ego"].tolist(),
                "object_3d_position": observation["object_3d_position"].tolist(),
            },
        }
        return np.array(self._send(request)["result"], dtype=np.float32)

    def get_server_metadata(self) -> dict:
        response = self._send({"action": "metadata"})
        return {
            "policy_name": response.get("policy_name"),
            "checkpoint": response.get("checkpoint"),
        }

    def infer_point_molmo(
        self, rgb: np.ndarray, object_name: str = None, prompt: str = None, task: str = "pick"
    ) -> np.ndarray:
        if object_name is None and prompt is None:
            raise ValueError("Either 'object_name' or 'prompt' must be provided")
        if object_name is not None and prompt is not None:
            raise ValueError("Provide either 'object_name' or 'prompt', not both")

        request = {
            "action": "infer_point",
            "rgb": rgb.tolist(),
        }

        if object_name is not None:
            request["object_name"] = object_name
            request["task"] = task
        else:
            request["prompt"] = prompt

        response = self._send(request)
        return np.array(response["point"], dtype=np.float32)

    def infer_point(self, rgb: np.ndarray, object_name: str, task: str) -> np.ndarray:
        try:
            import google.genai as genai
            from google.genai import types
        except ImportError as e:
            raise ImportError(
                "google-genai package is not installed. Please install it with `pip install google-genai`."
            ) from e

        if not hasattr(self, "_gemini_client"):
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY environment variable not set. "
                    "Please export GEMINI_API_KEY=your_api_key"
                )
            self._gemini_client = genai.Client(api_key=api_key)

        if rgb.dtype == np.float32 or rgb.dtype == np.float64:
            rgb = (rgb * 255).astype(np.uint8)
        image = Image.fromarray(rgb)

        if task == "pick":
            prompt = f"""Get the best point matching the following object: {object_name}. The label returned should be an identifying name for the object detected in the image, and the point should be at a point to pick that object.
    The answer should follow the json format: [{{"point": [y, x], "label": "{object_name}"}}...]. The points are in [y, x] format normalized to 0-1000."""
        elif task == "open":
            prompt = f"""Get the best point to be grasped to open the {object_name} (eg. a handle). The label returned should be an identifying name for the object detected in the image, and the point should be the best point to open that object.
    The answer should follow the json format: [{{"point": [y, x], "label": "{object_name}"}}...]. The points are in [y, x] format normalized to 0-1000."""
        elif task == "close":
            prompt = f"""Get the best point to be grasped to close the {object_name} (eg. a handle). The label returned should be an identifying name for the object detected in the image, and the point should be the best point to close that object.
    The answer should follow the json format: [{{"point": [y, x], "label": "{object_name}"}}...]. The points are in [y, x] format normalized to 0-1000."""
        else:
            raise ValueError(f"Unsupported task: {task}")

        try:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            image_bytes = buf.getvalue()

            contents = [
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/png",
                ),
                prompt,
            ]

            config = types.GenerateContentConfig(
                temperature=0.0, thinking_config=types.ThinkingConfig(thinking_budget=0)
            )

            response = self._call_gemini_api_with_retry(
                model="gemini-robotics-er-1.5-preview", contents=contents, config=config
            )
            generated_text = response.text

            json_point = self._parse_json_point_yx(generated_text)
            if json_point:
                y_pos, x_pos = json_point
                x_norm = float(x_pos) / 1000.0
                y_norm = float(y_pos) / 1000.0

                x_norm = max(0.0, min(1.0, x_norm))
                y_norm = max(0.0, min(1.0, y_norm))

                return np.array([x_norm, y_norm], dtype=np.float32)

            return np.array([0.5, 0.5], dtype=np.float32)

        except Exception as e:
            log.error(f"Gemini API error: {e}")
            return np.array([0.5, 0.5], dtype=np.float32)

    def _call_gemini_api_with_retry(self, model, contents, config, max_retries=3, base_delay=2):
        from google.genai import errors

        for attempt in range(max_retries):
            try:
                return self._gemini_client.models.generate_content(
                    model=model, contents=contents, config=config
                )
            except errors.ServerError as e:
                if e.code in [503, 429] and attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    error_msg = e.message if e.message else "Service unavailable"
                    log.warning(
                        f"API error {e.code}: {error_msg}. Retrying in {delay}s... (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                    continue
                else:
                    raise
            except Exception:
                raise

    def _parse_json_image_point(self, text):
        try:
            json_match = re.search(r"```(?:json)?\s*({.*?})\s*```", text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r"({.*?})", text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    return None

            data = json.loads(json_str)

            if "image_index" in data and "point" in data:
                image_idx = int(data["image_index"])
                point = data["point"]
                if isinstance(point, list) and len(point) >= 2:
                    return (image_idx, float(point[0]), float(point[1]))

            return None

        except (json.JSONDecodeError, KeyError, ValueError, IndexError) as e:
            log.error(f"JSON parsing error: {e}")
            return None

    def _parse_json_point_yx(self, text):
        try:
            json_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r"(\[.*?\])", text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    return None

            data = json.loads(json_str)
            if not isinstance(data, list) or len(data) == 0:
                return None

            point_data = data[0]
            if "point" in point_data:
                point = point_data["point"]
                if isinstance(point, list) and len(point) >= 2:
                    return (float(point[0]), float(point[1]))

            return None

        except (json.JSONDecodeError, KeyError, ValueError, IndexError) as e:
            log.error(f"JSON parsing error: {e}")
            return None

    def close(self):
        if self.websocket:
            self.websocket.close()
            self.websocket = None
