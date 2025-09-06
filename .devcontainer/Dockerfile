ARG BASE_IMAGE

FROM ${BASE_IMAGE}

ARG USERNAME=user
ARG USER_UID=1337
ARG USER_GID=$USER_UID
ARG BACKEND=cpu

# Delete existing user with UID 1000 if it exists then create a new user.
RUN if getent passwd 1000 > /dev/null 2>&1; then \
        EXISTING_USER=$(getent passwd 1000 | cut -d: -f1) && \
        userdel -r "$EXISTING_USER"; \
    fi \
    && groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME \
    && cp /etc/skel/.bashrc /home/$USERNAME/ \
    && cp /etc/skel/.profile /home/$USERNAME/ \
    && mkdir -p /home/$USERNAME/.cache /home/$USERNAME/.config \
    && chown -R $USERNAME:$USERNAME /home/$USERNAME \
    && echo 'PS1="$(tput setaf 241)[$BACKEND] $(tput bold)$(tput setaf 169)\w$(tput sgr0)\$ "' >> /home/$USERNAME/.bashrc

# Install any additional system dependencies if needed (e.g., git, curl)
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    jq curl ca-certificates file vim less python3-dev python3-full build-essential git cmake gcc-12 rsync nodejs npm \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu pip install --no-cache-dir uv llama-cpp-python bitsandbytes==0.46.1

# Install nvm and set up Node.js
RUN su - $USERNAME -c 'curl -o- "https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh" | bash' \
    && su - $USERNAME -c 'source ~/.nvm/nvm.sh && nvm install lts/iron && nvm alias default lts/iron'
    
# Set the working directory
WORKDIR /workspace