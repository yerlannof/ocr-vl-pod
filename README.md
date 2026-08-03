# ocr-vl-pod

Готовый образ GPU-пода для OWN-1: PaddleOCR-VL обучение (ERNIEKit v1.5 с
патчами save_model и LoRA-таргетов) + пайплайн-инференс (paddleocr 3.7.0).
Собирается на GitHub Actions → ghcr.io/yerlannof/ocr-vl-pod:v2.
Модель не вшита — snapshot_download при старте или network volume.
Использование: RunPod → Deploy → Custom image `ghcr.io/yerlannof/ocr-vl-pod:v2`.
