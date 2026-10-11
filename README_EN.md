# Comic Translator GUI (English → Arabic) - technical notes

Local desktop app: detect text → OCR (RapidOCR) → erase (LaMa ONNX / OpenCV) → translate
(Google / MyMemory / DeepL API) → typeset right-to-left Arabic with binary-search font fitting.
Arabic user guide: [README.md](README.md).

## Quick start

```bash
# Windows: double-click Start.bat      macOS/Linux: bash start.sh
# or manually:
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                       # minimal set, no PyTorch
python run.py                                         # GUI (auto-installs missing essentials)
python run.py cli page.png -o output                  # CLI
python run.py cli --download-models                   # LaMa ONNX -> models/lama_fp32.onnx
pytest -q
```

Linux needs Tk: `sudo apt install python3-tk`.

## Dependencies (kept deliberately small)

| Purpose | Package | Notes |
|---------|---------|-------|
| UI | customtkinter, tkinterdnd2 | drag & drop is optional |
| OCR | rapidocr + onnxruntime | models bundled (~33 MB) |
| Eraser | onnxruntime + LaMa ONNX | falls back to OpenCV inpainting |
| Translation | requests only | Google/MyMemory/DeepL via HTTP |
| Arabic | arabic-reshaper, python-bidi, Pillow | |

The Setup window (`src/ui/setup_dialog.py`, logic in `src/utils/installer.py`) checks every component and
installs missing ones with the current interpreter (`sys.executable -m pip`).

## Translation

`src/translators/`: `google.py` and `mymemory.py` use public HTTP endpoints directly; `deepl_api.py` is the official API;
Requests are sequential and throttled (default 0.5 s). Google receives the whole page in **one** request (one text per
line, split back by line; one-by-one fallback if the line count differs). A rate-limit (429) is never retried: the chain
fails over to the backup provider immediately. `TranslatorChain` skips providers that failed earlier in the session.
Credit/website/watermark blocks (`src/utils/text_filters.py`) are erased but not translated (`TextRegion.ignored`).
`diagnose()` (used by the "Test translation" buttons) tests each provider and explains failures.

## Layout

```
run.py / Start.bat / start.sh   launchers (auto-install)
src/main.py                     argparse entry (GUI, CLI)
src/config.py                   settings + paths
src/core/{models,detector,ocr,inpainter,renderer,pipeline}.py
src/translators/{base,google,mymemory,deepl_api}.py
src/utils/{arabic_utils,color_utils,font_matcher,image_utils,installer,text_filters}.py
src/ui/{app,canvas,widgets,setup_dialog}.py
tests/                          pytest suite
```
