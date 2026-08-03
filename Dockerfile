# OWN-1: готовый образ GPU-пода для PaddleOCR-VL (обучение + пайплайн-инференс).
# Цель: под из образа готов к работе за ~2 минуты в любом регионе RunPod —
# вместо 15-25 минут бутстрапа окружения на каждом поде.
#
# База = тот же образ, что в наших боевых подах (ssh/init RunPod проверены).
# Внутри: ОДИН venv /opt/venv (paddle-gpu cu126 + ERNIEKit + paddleocr) +
# клон ERNIE с двумя нашими патчами (save_model kwarg, LoRA-таргеты под
# реальные имена модулей PaddleOCR-VL). Модель НЕ вшита (2 ГБ) — качается
# при старте за 1-2 минуты дата-центровым каналом или берётся с тома.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends rsync git \
    && rm -rf /var/lib/apt/lists/*

# единый venv: paddle-gpu + инференс-стек (версии = боевые с 3080)
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install -U pip \
    && /opt/venv/bin/pip install paddlepaddle-gpu==3.2.1 \
         -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ \
    && /opt/venv/bin/pip install paddleocr==3.7.0 paddlex==3.7.2 safetensors \
    && /opt/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu

# ERNIEKit (обучение) в тот же venv
RUN git clone --depth 1 -b release/v1.5 https://github.com/PaddlePaddle/ERNIE /opt/ERNIE \
    && /opt/venv/bin/pip install -r /opt/ERNIE/requirements/gpu/requirements.txt \
    && /opt/venv/bin/pip install -e /opt/ERNIE \
    && /opt/venv/bin/pip install huggingface_hub

# патч 1: save_model под paddleformers 0.4 (выстрадан в v1)
RUN sed -i 's/def save_model(self, output_dir=None):/def save_model(self, output_dir=None, merge_tensor_parallel=False):/' \
        /opt/ERNIE/erniekit/train/ocr_vl_sft/pretraining_trainer.py \
    && mkdir -p /opt/ERNIE/download_tmp/raw_video

# патч 2: LoRA-таргеты = реальные имена модулей PaddleOCR-VL
# (дефолтные fused-имена не матчат модель — в v1 обучились 36 пар из 126)
COPY patch_lora_targets.py /opt/
RUN python3 /opt/patch_lora_targets.py /opt/ERNIE && rm /opt/patch_lora_targets.py

LABEL org.opencontainers.image.source="https://github.com/yerlannof/ocr-vl-pod" \
      org.opencontainers.image.description="PaddleOCR-VL train+infer pod (ERNIEKit v1.5 patched)"
