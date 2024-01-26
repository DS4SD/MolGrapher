FROM python:3.10-slim-bullseye AS base

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get -y update && apt-get install -y \
    procps \
    libglib2.0-0 libgl1-mesa-glx libsm6 libxext6 \
    curl build-essential git libssl-dev wget unzip \
    libxrender-dev libcairo2-dev

ENV PATH="${PATH}:/app"

FROM base AS dependency

# Create user
RUN useradd -ms /bin/bash app \
    && mkdir /app \
    && chown -R app:0 /app \
    && chmod g=u -R /app

WORKDIR /app

# Pip install
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    pip install --no-warn-script-location torch==1.12.1+cpu torchvision==0.13.1+cpu -f https://download.pytorch.org/whl/torch_stable.html  \
    && pip install --no-warn-script-location pytorch_lightning==2.0.2 torch-geometric==2.3.1 \
    scikit-learn seaborn timm mahotas more_itertools

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    pip install --no-warn-script-location rdkit-pypi CairoSVG SmilesPE python-Levenshtein \
    nltk ipykernel ipython rouge-score opencv-python \
    albumentations paddleocr paddlepaddle \
    torchsummary weighted-levenshtein

FROM dependency AS specific

WORKDIR /app

# Copy repositories
COPY --chown=app:0 . /app/MolGrapher/

# Pip install 
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
       pip install -e /app/MolGrapher/ \
    && pip install -e /app/MolGrapher/MolDepictor/ 

FROM specific AS production

WORKDIR /app

# Download model
RUN wget https://huggingface.co/ds4sd/MolGrapher/resolve/main/models/graph_classifier/gc_gcn_model.ckpt -O /app/MolGrapher/data/models/graph_classifier/gc_gcn_model.ckpt
RUN wget https://huggingface.co/ds4sd/MolGrapher/resolve/main/models/graph_classifier/gc_no_stereo_model.ckpt -O /app/MolGrapher/data/models/graph_classifier/gc_no_stereo_model.ckpt
RUN wget https://huggingface.co/ds4sd/MolGrapher/resolve/main/models/graph_classifier/gc_stereo_model.ckpt -O /app/MolGrapher/data/models/graph_classifier/gc_stereo_model.ckpt
RUN wget https://huggingface.co/ds4sd/MolGrapher/resolve/main/models/keypoint_detector/kd_model.ckpt -O /app/MolGrapher/data/models/keypoint_detector/kd_model.ckpt

RUN chmod ugo+rwx -R /app
WORKDIR /app/MolGrapher/molgrapher/scripts/annotate/

CMD ["/bin/sh", "./run.sh"]