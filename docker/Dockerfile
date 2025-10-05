ARG BACKEND=cpu
ARG USERNAME=user
ARG USER_ID=1000
ARG DEVICE=auto

FROM ubuntu:24.04 AS base
ARG USER_ID
ARG USERNAME
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NODE_ENV=production \
    NVM_DIR=/opt/nvm \
    PIP_BREAK_SYSTEM_PACKAGES=1
# Delete existing user with UID $USER_ID if it exists then create a new user.
RUN if getent passwd $USER_ID > /dev/null 2>&1; then \
        EXISTING_USER=$(getent passwd $USER_ID | cut -d: -f1) && \
        userdel -r "$EXISTING_USER"; \
    fi \
    && groupadd --gid $USER_ID $USERNAME \
    && getent group render || groupadd render \
    && getent group video || groupadd video \
    && useradd --uid $USER_ID --gid $USER_ID -G render,video -m -s /bin/bash $USERNAME \
    && mkdir -p /home/$USERNAME/.cache /home/$USERNAME/.config \
    && chown -R $USERNAME:$USERNAME /home/$USERNAME

# Install node, upgrade pip, install uv.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash python3-full python3-setuptools python3-pip python-is-python3 dumb-init curl build-essential make g++ \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir uv \
    && pip install --upgrade --no-cache-dir \
        uv packaging \
    && mkdir -p -m 655 $NVM_DIR \
    && curl -o- "https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh" | bash \
    && . "$NVM_DIR/nvm.sh" && nvm install lts/jod && nvm alias default lts/jod

### Runtime bases per backend ###
# CPU base
FROM base AS runtime-cpu
RUN pip3 install torch --no-cache-dir --index-url https://download.pytorch.org/whl/cpu

# CUDA base
FROM base AS runtime-cuda
RUN pip3 install torch --no-cache-dir --index-url https://download.pytorch.org/whl/cu126

# ROCm base
FROM base AS runtime-rocm
RUN pip install --no-cache-dir \
        'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0/pytorch_triton_rocm-3.4.0%2Brocm7.0.0.gitf9e5bf54-cp312-cp312-linux_x86_64.whl' \
        'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0/torch-2.8.0%2Brocm7.0.0.git64359f59-cp312-cp312-linux_x86_64.whl'

### selector##
FROM runtime-${BACKEND} AS runtime
ARG DEVICE
ARG USER_ID
# Copy Python project
COPY --chmod=655 py/pyproject.toml py/uv.lock /app/py/
COPY --chmod=655 py/src /app/py/src/
# Copy TS project
COPY --chmod=655 js/ /app/js/

ENV UV_LINK_MODE=copy \
    DEVICE=${DEVICE:-auto}
# Install setup py and js projects
RUN . "$NVM_DIR/nvm.sh" && \
    cd /app/py && uv venv --system-site-packages && uv --no-cache sync --all-extras --frozen && \
    cd /app/js && npm ci --include dev && npm run build && rm -rf src

ARG EXPOSE_PORT=8771
EXPOSE ${EXPOSE_PORT}

# Entrypoint that can run asr | bot | both
COPY --chmod=655 docker/entrypoint.sh /app/entrypoint.sh
RUN chmod a+x /app /app/py /app/js

USER $USER_ID
WORKDIR /app
ENTRYPOINT ["dumb-init", "--", "/bin/sh", "-c", ". \"$NVM_DIR/nvm.sh\" && exec /app/entrypoint.sh"]
