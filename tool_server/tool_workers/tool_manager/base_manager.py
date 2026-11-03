"""HTTP-only adapter for the five online workers; no offline tools or global signals."""
import requests
from PIL import Image
from io import BytesIO
import base64


class ToolManager:
    def __init__(self, controller_url_location='http://localhost:20001', tools=None):
        self.controller_addr = controller_url_location.rstrip('/')
        self.session = requests.Session()
        self.session.trust_env = False
        self.tools = tools
        registered = self._post(self.controller_addr + '/list_models', {})['models']
        self.available_online_tools = [name for name in registered if tools is None or name in tools]
        missing = set(tools or []) - set(self.available_online_tools)
        if missing:
            raise RuntimeError(f'Tools missing from controller: {sorted(missing)}')
        self.online_tool_addr_dict = {}
        for name in self.available_online_tools:
            self.online_tool_addr_dict[name] = self._post(
                self.controller_addr + '/get_worker_address', {'model': name})['address']

    def _post(self, url, payload):
        response = self.session.post(url, json=payload, timeout=(10, 300))
        response.raise_for_status()
        return response.json()

    def call_tool(self, tool_name, params):
        if tool_name not in self.online_tool_addr_dict:
            return {'text': f'Tool {tool_name} not found.', 'error_code': 1}
        try:
            result = self._post(self.online_tool_addr_dict[tool_name].rstrip('/') + '/worker_generate', params)
            edited = result.get('edited_image')
            if edited:
                image = Image.open(BytesIO(base64.b64decode(edited)))
                if min(image.size) < 28:
                    result.pop('edited_image')
            return result
        except Exception as error:
            return {'text': f'Failed to call tool {tool_name}: {error}', 'error_code': 1}

