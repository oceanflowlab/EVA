# Third-Party Code and Assets

## VLMEvalKit

- Upstream: https://github.com/open-compass/VLMEvalKit
- Local upstream base: `0a11caf2253a1cdaa711deca30dca7a7d38da7e1`.
- License: Apache-2.0; original notice and full license are retained under `licenses/`.
- This repository vendors Qwen image inference, InternVL image preprocessing,
  native benchmark prompts/scorers, and their utility dependencies. Registry/entry
  points are scoped to the two backbones and four datasets.
- EVA changes include evidence routing, iterative tools, evidence compression,
  dataset-specific direct-answer prompts, JSON traces, and release launchers.

## AdaReasoner

- Upstream: https://github.com/ssmisya/AdaReasoner
- Local upstream base: `c21a654a75a3a075cc4c3b5db3863c54e4464aa5`.
- Files: `tool_server/tool_workers/online_workers/` and most `tool_server/utils/`.
- The local upstream checkout has **no top-level LICENSE file**. Its `setup.py`
  advertises `License :: OSI Approved :: MIT License`; that metadata is preserved
  verbatim in `licenses/AdaReasoner-setup.py.txt`, but is not a complete license grant.
- **Before publishing this directory, obtain/verify the upstream license text and
  copyright notice, or replace the affected worker code.** Do not assume the EVA
  Apache-2.0 license relicenses those files. This local release candidate is not a
  representation that the missing upstream authorization has been resolved.
- The HTTP-only tool manager and process launcher were written for the EVA release.

## External Models and Datasets

Qwen2.5-VL, InternVL3, GroundingDINO, PaddleOCR, PaddleOCR-VL and the evaluation datasets
retain their own licenses and terms. Weights, benchmark data, downloaded source
trees, experiment outputs, and private credentials are not distributed here.
The tool smoke test bundles EVA-generated text/chart fixtures and NASA's
public-domain astronaut image, distributed via scikit-image. Attribution and
image descriptions are in `tool_service_tests/fixtures/README.md`.
