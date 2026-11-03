"""Element-level chart recognition using the official PaddleOCR-VL HF model."""

import threading

from PIL import Image


class PaddleOCRVLChartBackend:
    def __init__(self, model_path='PaddlePaddle/PaddleOCR-VL-1.6', device='cuda',
                 torch_dtype='auto', attn_implementation='sdpa', revision=None,
                 max_new_tokens=2048, max_pixels=1280 * 28 * 28):
        import torch
        import transformers
        from packaging.version import Version
        if Version(transformers.__version__).major < 5:
            raise RuntimeError('PaddleOCR-VL-1.6 requires Transformers >=5. '
                               'Start GetBarInfo in its isolated getbar-vl environment.')
        from transformers import AutoModelForImageTextToText, AutoProcessor
        if max_new_tokens <= 0 or max_pixels <= 0:
            raise ValueError('max_new_tokens and max_pixels must be positive')
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested for GetBarInfo but is not available')
        if torch_dtype == 'auto':
            dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.startswith('cuda') else torch.float32
        else:
            if torch_dtype not in {'bfloat16', 'float16', 'float32'}:
                raise ValueError('Unsupported torch_dtype')
            dtype = getattr(torch, torch_dtype)
        kwargs = {'revision': revision} if revision else {}
        self.processor = AutoProcessor.from_pretrained(model_path, **kwargs)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=dtype, attn_implementation=attn_implementation, **kwargs,
        ).to(device).eval()
        self.model_id = model_path
        self.revision = revision or getattr(self.model.config, '_commit_hash', None)
        self.max_new_tokens = max_new_tokens
        self.max_pixels = max_pixels
        self._lock = threading.Lock()

    def predict_chart(self, image):
        import torch
        if not isinstance(image, Image.Image):
            raise TypeError('Expected a PIL image')
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'image': image.convert('RGB')},
            {'type': 'text', 'text': 'Chart Recognition:'},
        ]}]
        with self._lock, torch.inference_mode():
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors='pt',
                images_kwargs={'size': {'shortest_edge': self.processor.image_processor.size['shortest_edge'],
                                       'longest_edge': self.max_pixels}},
            ).to(self.model.device)
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            generated = outputs[0][inputs['input_ids'].shape[-1]:]
            eos = self.model.generation_config.eos_token_id
            eos_ids = set(eos if isinstance(eos, list) else [eos])
            if len(generated) >= self.max_new_tokens and int(generated[-1]) not in eos_ids:
                raise RuntimeError('Chart output was truncated; increase max_new_tokens')
            text = self.processor.decode(generated, skip_special_tokens=True).strip()
            if not text:
                raise RuntimeError('PaddleOCR-VL returned an empty chart result')
        return {'result': text, 'model_id': self.model_id, 'model_revision': self.revision,
                'generated_tokens': len(generated)}
