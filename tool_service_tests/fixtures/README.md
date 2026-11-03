# Test Images

| File | Purpose | Source |
| --- | --- | --- |
| `ocr.png` | Read `EVA TOOL TEST` and `TOTAL 12345` | Generated for EVA; Apache-2.0 |
| `bar_chart.png` | Extract Alpha=20, Beta=40, Gamma=60 | Generated for EVA; Apache-2.0 |
| `astronaut.png` | Detect a person and test drawing annotations | NASA, via scikit-image |

The photograph depicts astronaut Eileen Collins and comes from the NASA Great
Images collection. It is distributed as a public-domain sample by
[scikit-image](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut).
These images are bundled with the repository; the smoke test runs without an
image download or a system font dependency.
