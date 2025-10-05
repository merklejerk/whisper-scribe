ARG USERNAME=user
ARG USER_ID=1000
ARG BACKEND=cpu
ARG ROCM_ARCH=gfx1100

FROM ubuntu:24.04 AS base
ARG USERNAME
ARG USER_ID
ARG BACKEND
ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    DEBIAN_FRONTEND=noninteractive \
    NVM_DIR=/opt/nvm
# Install common deps.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-full python3-setuptools python3-wheel python3-pip \
        python-is-python3 make g++ build-essential jq curl wget rsync \
        git ffmpeg bash net-tools file cmake make vim less xxd \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade --no-cache-dir \
        setuptools wheel uv llama-cpp-python packaging
# Delete existing user with UID $USER_ID if it exists then create a new user.
RUN if getent passwd $USER_ID > /dev/null 2>&1; then \
        EXISTING_USER=$(getent passwd $USER_ID | cut -d: -f1) && \
        userdel -r "$EXISTING_USER"; \
    fi \
    && groupadd --gid $USER_ID $USERNAME \
    && getent group render || groupadd render \
    && getent group video || groupadd video \
    && useradd --uid $USER_ID --gid $USER_ID -G render,video -m -s /bin/bash $USERNAME \
    && cp /etc/skel/.bashrc /home/$USERNAME/ \
    && cp /etc/skel/.profile /home/$USERNAME/ \
    && mkdir -p /home/$USERNAME/.cache /home/$USERNAME/.config \
    && chown -R $USERNAME:$USERNAME /home/$USERNAME \
    && echo 'PS1="$(tput setaf 241)[$BACKEND] $(tput bold)$(tput setaf 169)\w$(tput sgr0)\$ "' >> /home/$USERNAME/.bashrc \
    && echo 'source ${NVM_DIR}/nvm.sh' >> /home/$USERNAME/.bashrc

# Install nvm and set up Node.js
RUN chmod a+rx $NVM_DIR \
    && curl -o- "https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh" | bash \
    && source "$NVM_DIR/nvm.sh" && nvm install lts/jod && nvm alias default lts/jod

### Runtime bases per backend ###
# CPU base (Python 3.12 slim + PyTorch CPU wheels)
FROM base AS runtime-cpu
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --nocache-dir

# CUDA base (Python 3.12 slim + PyTorch CUDA wheels)
FROM base AS runtime-cuda
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --no-cache-dir \
    && pip install --upgrade --no-cache-dir bitsandbytes==0.46.1

# ROCm base (Python 3.12 slim + PyTorch ROCm wheels + ROCm libraries + bitsandbytes from source)
FROM base AS runtime-rocm
ARG ROCM_ARCH
WORKDIR /root
COPY bnb.patch .
RUN pip install --no-cache-dir \
        'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0/pytorch_triton_rocm-3.4.0%2Brocm7.0.0.gitf9e5bf54-cp312-cp312-linux_x86_64.whl' \
        'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0/torch-2.8.0%2Brocm7.0.0.git64359f59-cp312-cp312-linux_x86_64.whl' \
        'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0/torchaudio-2.8.0%2Brocm7.0.0.git6e1c7fe9-cp312-cp312-linux_x86_64.whl' \
        'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0/torchvision-0.24.0%2Brocm7.0.0.gitf52c4f1a-cp312-cp312-linux_x86_64.whl' \
    # Install ROCm libraries
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates gpg \
    && mkdir -p --mode=0755 /etc/apt/keyrings \
    && curl 'https://repo.radeon.com/rocm/rocm.gpg.key' \
        | gpg --dearmor | tee /etc/apt/keyrings/rocm.gpg > /dev/null \
    && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.0.1 noble main" \
        | tee /etc/apt/sources.list.d/rocm.list \
    && echo 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' \
        | tee /etc/apt/preferences.d/rocm-pin-600 \
    && apt update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        rocm \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
# Build bitsandbytes from source with ROCm support
RUN git clone --depth 1 --branch 0.47.0 https://github.com/bitsandbytes-foundation/bitsandbytes.git \
    && pwd && ls -la \
    && git apply --directory bitsandbytes/ bnb.patch \
    && cd bitsandbytes/ \
    && cmake -DGPU_TARGETS="$ROCM_ARCH" -DBNB_ROCM_ARCH="$ROCM_ARCH" -DCOMPUTE_BACKEND=hip -DwarpSize=32 -S . --fresh \
    && make \
    && pip install --no-cache-dir --no-deps . \
    && cd .. \
    && rm -rf bitsandbytes

### Backend selector ###
FROM runtime-${BACKEND} AS runtime
WORKDIR /workspace