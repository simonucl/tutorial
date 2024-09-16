HUGGINGFACE_TOKEN=""
WANDB_API_KEY=""
GH_TOKEN=""

# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda

# Initialize conda for bash shell
eval "$($HOME/miniconda/bin/conda shell.bash hook)"

# Create conda environment
conda create -y -n chats python=3.10

# Activate the environment
conda activate chats

# Clean up
rm miniconda.sh

# Install GitHub CLI
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh -y

apt-get update && apt-get install -y git && apt-get install -y vim

# Install pip packages
pip install -r requirements.txt

# Login to HuggingFace and Weights & Biases
# huggingface-cli login --token $HUGGINGFACE_TOKEN
# wandb login --api-key $WANDB_API_KEY
