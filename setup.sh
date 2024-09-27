#!/bin/bash

setup_git() {
    read -p "Enter GitHub email: " github_email
    read -sp "Enter GitHub auth token: " github_token
    echo

    git config --global user.email "$github_email"
    
    # Store the token securely
    echo "https://oauth2:${github_token}@github.com" > ~/.git-credentials
    git config --global credential.helper store

    # Test the token
    if curl -s -H "Authorization: token $github_token" https://api.github.com/user | grep -q "login"; then
        echo "GitHub authentication successful."
    else
        echo "GitHub authentication failed. Please check your token and try again."
        exit 1
    fi

    read -p "Enter GitHub repository URL: " repo_url

    git clone "$repo_url"
    
    # Get the repository name from the URL
    repo_name=$(basename "$repo_url" .git)
    
    # Move the current script into the cloned repository
    mv "$0" "$repo_name/"
    
    echo "Success"
}

install_rye() {
    curl -sSf https://rye.astral.sh/get | bash
    if ! echo 'source "$HOME/.rye/env"' >> ~/.profile 2>/dev/null; then
        echo "Failed to update ~/.profile."
    fi
    if ! echo 'source "$HOME/.rye/env"' >> ~/.bashrc 2>/dev/null; then
        echo "Failed to update ~/.bashrc."
    fi
    source "$HOME/.rye/env"
    rye sync
}

check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        echo "CUDA GPU available:"
        nvidia-smi
        python -c "import torch; print('PyTorch CUDA available:', torch.cuda.is_available())"
    else
        echo "CUDA GPU not found"
    fi
}

case "$1" in
    "git") setup_git ;;
    "rye") install_rye ;;
    "gpu") check_gpu ;;
    *) echo "Usage: $0 {git|rye|gpu}" ;;
esac