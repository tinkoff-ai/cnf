apt update
apt upgrade
apt install tmux nano
pip install wandb scikit-learn scipy gym==0.23.0 matplotlib seaborn
bash ./install_mujoco.sh
git config --global --add safe.directory /flows
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin" >> ~/.bashrc
echo "export PYTHONPATH=." >> ~/.bashrc
