"""
A GetBar worker with PaddleOCR-VL-1.6 and an explicit legacy PP-Chart2Table option.
"""

import base64
import os
import sys
import tempfile
import traceback
import uuid
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image
from transformers import HfArgumentParser

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.error_codes import *
from tool_server.utils.server_utils import build_logger
from tool_server.utils.worker_arguments import WorkerArguments

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"get_bar_worker_{worker_id}.log")


@dataclass
class GetBarArguments(WorkerArguments):
    backend: str = field(default='paddleocr_vl', metadata={"help": "paddleocr_vl or pp_chart2table"})
    model_revision: Optional[str] = None
    max_new_tokens: int = 2048
    max_pixels: int = 1280 * 28 * 28
    torch_dtype: str = 'auto'
    attn_implementation: str = 'sdpa'
    max_concurrency: int = field(
        default=1,
        metadata={"help": "Maximum number of concurrent requests the model can handle"}
    )


class GetBarWorker(BaseToolWorker):
    def __init__(self, worker_arguments: GetBarArguments = None):
        if worker_arguments and worker_arguments.model_name is None:
            worker_arguments.model_name = "GetBarInfo"

        if worker_arguments:
            worker_arguments.limit_model_concurrency = worker_arguments.max_concurrency

        super().__init__(worker_arguments)

        self.instruction = {
            "type": "function",
            "function": {
                "name": "GetBarInfo",
                "description": "Extract chart labels, series and values. Returns parsed chart evidence.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "Base64 encoded image string."
                        }
                    },
                    "required": ["image"]
                }
            }
        }

    def init_model(self):
        self.backend = self.args.backend
        if self.backend == 'paddleocr_vl':
            from tool_server.tool_workers.online_workers.chart_vl_backend import PaddleOCRVLChartBackend
            self.model = PaddleOCRVLChartBackend(
                model_path=self.model_path or 'PaddlePaddle/PaddleOCR-VL-1.6',
                device=self.device, torch_dtype=self.args.torch_dtype,
                attn_implementation=self.args.attn_implementation,
                revision=self.args.model_revision, max_new_tokens=self.args.max_new_tokens,
                max_pixels=self.args.max_pixels,
            )
        elif self.backend == 'pp_chart2table':
            from paddlex import create_model
            self.model = create_model('PP-Chart2Table')
        else:
            raise ValueError(f'Unknown GetBarInfo backend: {self.backend}')
        logger.info(f"Initialized {self.model_name} backend={self.backend}")

    def get_tool_instruction(self):
        return self.instruction

    def _to_jsonable(self, obj: Any):
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            return {str(k): self._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_jsonable(x) for x in obj]

        for method_name in ("to_dict", "dict"):
            if hasattr(obj, method_name) and callable(getattr(obj, method_name)):
                try:
                    return self._to_jsonable(getattr(obj, method_name)())
                except Exception:
                    pass

        if hasattr(obj, "__dict__"):
            try:
                return self._to_jsonable(vars(obj))
            except Exception:
                pass

        return str(obj)

    def _decode_image_to_tempfile(self, image_b64: str) -> str:
        image = Image.open(BytesIO(base64.b64decode(image_b64))).convert("RGB")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name
        image.save(tmp_path)
        return tmp_path

    def generate(self, params: Dict[str, Any]):
        tool_reward = 2.0

        param_keys = set(params.keys())
        required_keys = set(self.instruction["function"]["parameters"]["required"])
        parameter_name_match_reward = len(param_keys & required_keys) / len(required_keys | param_keys)
        tool_reward += parameter_name_match_reward

        if parameter_name_match_reward < 1:
            return {
                "tool_response_from": self.model_name,
                "status": "failed",
                "message": "Invalid parameters: expected keys: image.",
                "error_code": INVALID_PARAMETERS,
                "tool_reward": tool_reward,
            }

        image_path = None
        try:
            image_b64 = params["image"]
            if self.backend == 'paddleocr_vl':
                image = Image.open(BytesIO(base64.b64decode(image_b64))).convert('RGB')
                chart_results = [self.model.predict_chart(image)]
            else:
                image_path = self._decode_image_to_tempfile(image_b64)
                results = self.model.predict(input={"image": image_path}, batch_size=1)
                chart_results = [self._to_jsonable(res) for res in results]

            return {
                "tool_response_from": self.model_name,
                "status": "success",
                "message": f"Chart parsed successfully by {self.backend}.",
                "backend": self.backend,
                "chart_results": chart_results,
                "error_code": SUCCESS,
                "tool_reward": tool_reward + 1.0,
            }

        except Exception as e:
            logger.error(f"Error in GetBar: {e}")
            logger.error(traceback.format_exc())
            return {
                "tool_response_from": self.model_name,
                "status": "failed",
                "message": f"Error: {str(e)}",
                "error_code": TOOL_RUN_FAILED,
                "tool_reward": tool_reward,
            }
        finally:
            if image_path and os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except Exception:
                    pass


if __name__ == "__main__":
    parser = HfArgumentParser((GetBarArguments,))
    args, = parser.parse_args_into_dataclasses()

    logger.info(f"args: {args}")

    worker = GetBarWorker(worker_arguments=args)
    worker.run()
