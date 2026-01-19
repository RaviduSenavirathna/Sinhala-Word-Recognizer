# Sinhala Handwriting Checker — Edge-Lite (MobileNetV3 + 5-Rule Feedback)

Tiny, explainable pipeline to check **Grade 2–3 Sinhala** handwriting on **5-rule** paper.  
It segments a word, classifies letters/graphemes, and gives simple **“on the line”** feedback.  
Designed to run **in Google Colab** and export to **TFLite** for mobile/web.

---

## ✨ Features

- **Edge-Lite classifier**: MobileNetV3-Small @ **128×128**, grayscale→RGB  
- **Data-light** friendly (works with small datasets + augmentations)  
- **Connected Components** segmentation (OpenCV) for word crops  
- **5-rule checks**: baseline band compliance, quick overshoot flags  
- **Two-phase training**: head warmup → partial unfreeze fine-tune  
- **Keras 3 saving**: `.keras` model file + `SavedModel` export  
- **TFLite**: dynamic-range + optional **INT8** quantization  
- **Gradio demo**: upload an image → JSON results + visual overlay  
- Optional next step: **multi-head pili** model (type + zone)

---
