# Third-party software and data

RiskSifter's original source code is MIT-licensed. Dependency licenses remain
independent; installation downloads those components separately.

| Component | License / source |
|---|---|
| FastAPI | MIT — https://github.com/fastapi/fastapi |
| Starlette / Uvicorn | BSD-3-Clause — https://github.com/encode/starlette / https://github.com/encode/uvicorn |
| OpenCV | Apache-2.0 — https://github.com/opencv/opencv |
| NumPy | BSD-3-Clause — https://github.com/numpy/numpy |
| Pillow | HPND — https://github.com/python-pillow/Pillow |
| imageio-ffmpeg Python wrapper | BSD-2-Clause — https://github.com/imageio/imageio-ffmpeg |
| FFmpeg executable | License depends on build flags. The installed libx264-enabled executable may be GPL. Invoked as a separate subprocess; not copied into this source repository. Inspect with `ffmpeg -L`. |
| PyTorch / torchvision (optional) | BSD-style — https://github.com/pytorch/pytorch |
| Transformers / Accelerate (optional) | Apache-2.0 — https://github.com/huggingface/transformers |
| SmolVLM2 model weights (optional) | Apache-2.0 — https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct |
| comma2k19 data | MIT, Copyright (c) 2018 comma.ai — https://github.com/commaai/comma2k19/blob/master/LICENSE |
| TUM Traffic Accid3nD data | CC BY-NC-SA 4.0, Zimmer et al. — https://accident-dataset.github.io/ |

No code was copied from AutoMine, K-Risk, Simple-TAD, DashLab or DoTA. Their
designs were researched, and citations are in docs/TECHNICAL_DECISIONS.md.

The data downloader preserves attribution and provenance. Raw media, weights,
environments and generated dataset packages are not included in the source
distribution. Exporting a clip does not change its source license. In particular,
TUM media/derived clips must not be relicensed as MIT or offered for commercial use.

comma.ai MIT notice, applicable to its downloaded sample:

Copyright (c) 2018 comma.ai

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
