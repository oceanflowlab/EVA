# Tool Environments

The worker code is retained from the experiment workspace with a minimal HTTP
transport and portable launcher. No model weights are included. CUDA-compatible
PyTorch and PaddlePaddle wheels must match the host driver/GPU. Existing
`tool-server`, OCR-v6 and GetBarInfo environments can also be used by passing
their Python executables explicitly.

## Shared Worker Environment

```bash
conda create -n eva-tools python=3.10 -y
conda activate eva-tools
# Install a CUDA-compatible torch/torchvision pair, then:
pip install -r requirements/tools.txt
git clone https://github.com/IDEA-Research/GroundingDINO.git third_party/GroundingDINO
pip install --no-build-isolation -e third_party/GroundingDINO
```

Build GroundingDINO's extension against the installed PyTorch and CUDA toolkit.
The GroundingDINO repository and weights have their own licenses. The worker
expects the Swin-T OGC checkpoint and its matching config; supply both paths to
the launcher. The downloaded checkout is ignored by Git.

## OCR

```bash
conda create -n eva-ocr python=3.10 -y
conda activate eva-ocr
pip install -r requirements/ocr.txt
# Install paddlepaddle-gpu from the official wheel index for your CUDA version.
```

The configured models are **PP-OCRv6_medium_det + PP-OCRv6_medium_rec**.
The locally observed environment was PaddleOCR 3.7.0 / PaddlePaddle-GPU 3.3.0,
torch 2.6.0+cu124 / Transformers 4.57.6. These observed versions are not a claim
that every CUDA platform is supported. The Paddle model cache/download source
can be configured using `PADDLE_PDX_CACHE_HOME` / `PADDLE_PDX_MODEL_SOURCE`.

## GetBarInfo

```bash
conda create -n eva-chart python=3.10 -y
conda activate eva-chart
pip install -r requirements/chart.txt
```

Uses **PaddlePaddle/PaddleOCR-VL-1.6**, revision
`c5630abae1d940eafe0697512a0325494b02ab42`, with `Chart Recognition:`,
SDPA, BF16/FP16 on CUDA, max_new_tokens=2048, max_pixels=1003520. It returns
`chart_results[].result`, not a new tool schema. Transformers >=5 is required;
do not upgrade the Qwen environment to >=5 for this tool. The local observed
chart environment was Transformers 5.19.0 / torch 2.6.0+cu124.

## Ports and Network

Default controller: 20001; OCR: 20009; GroundingDINO: 20010; DrawLine: 20013;
DrawShape: 20014; GetBarInfo: 20017. `--port` shifts the entire block.
Workers have no authentication and must not be exposed to the public Internet.
The launcher binds `127.0.0.1`; HTTP tool requests bypass proxy environment
variables. A configured Hugging Face proxy/mirror is used only for downloads.

Set `HF_ENDPOINT` explicitly only when needed; this repository does not force
an unofficial mirror. `--dry-run` prints commands without launching processes.
Startup errors are fatal; the launcher does not silently continue without tools.

## Check All Five Tools

```bash
python tool_service_tests/test_five_tools.py \
  --controller http://localhost:20001 --timeout 300
```

The repository includes text, chart, and object-detection test images. The command
prints a PASS/FAIL row for each tool and an overall result. It checks recognized
text, chart labels/values, detection boxes, and edited images, then saves the
responses under `outputs/tool_service_tests/`. The exit code is 0 when all five
tools pass and 1 if any tool fails.
