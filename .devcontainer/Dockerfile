FROM nvidia/cuda:11.6.0-devel-ubuntu20.04

ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

ENV PATH=${PATH}:/root/.local/bin
ENV HISTFILE=/root/.history/.zsh_history

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update \
    && apt -y install --no-install-recommends apt-utils 2>&1

RUN apt -y dist-upgrade && \
    apt install --no-install-recommends -y \
        build-essential \
        git \
        git-lfs \
        gnupg-agent \
        groff \
        curl \
        dialog \
        wget \
        unzip \
        openssh-client \
        python3.9 \
        python3.9-dev \
        python3.9-venv \
        python3-pip \ 
        libyaml-dev \
        apt-transport-https \
        software-properties-common \
        xz-utils \
        zsh \
        pkg-config \
        libhdf5-dev \
        graphviz \
    && apt -y autoremove && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Prepare git lfs in case we need it
RUN git lfs install --skip-smudge

# set zsh as the defaul shell
RUN chsh -s $(which zsh)

# Install Python tools
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
RUN python3.9 -m pip install --no-cache-dir -U pip wheel setuptools

# Install poetry in preview mode to be able to set the correct version of setuptools afterwards.
RUN curl -sSL https://install.python-poetry.org | python3.9 -

ARG WORKSPACE_FOLDER
RUN mkdir /workspace
WORKDIR ${WORKSPACE_FOLDER}

# run the oh-my-zsh installation script  
RUN wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true
RUN sed -i 's/plugins=(git)/plugins=(git poetry)/g' $HOME/.zshrc
RUN sed -i 's/ZSH_THEME="robbyrussell"/ZSH_THEME="kolo"/g' $HOME/.zshrc

# install powerlevel10k
# RUN git clone https://github.com/romkatv/powerlevel10k.git ~/.oh-my-zsh/custom/themes/powerlevel10k

# RUN curl -fsSLO https://raw.githubusercontent.com/romkatv/dotfiles-public/master/.purepower

ENV DEBIAN_FRONTEND=dialog
