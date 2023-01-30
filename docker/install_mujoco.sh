apt-get update -q
apt-get install -y \
  wget \
  curl \
  git \
  libgl1-mesa-dev \
  libgl1-mesa-glx \
  libglew-dev \
  libosmesa6-dev \
  software-properties-common \
  net-tools \
  vim \
  build-essential

mkdir -p /root/.mujoco \
  && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
  && tar -xf mujoco.tar.gz -C /root/.mujoco \
  && rm mujoco.tar.gz

pip install mujoco-py
pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl
