#!/bin/bash

set -e  # Exit on error

# 🔧 Update system and install build dependencies
echo "Installing build dependencies..."
sudo apt update
sudo apt install -y \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libgdbm-dev \
    libdb5.3-dev \
    libbz2-dev \
    libexpat1-dev \
    liblzma-dev \
    tk-dev \
    libffi-dev \
    wget \
    curl \
    git \
    uuid-dev

# 📦 Define Python version to install
PYTHON_VERSION="3.11.6"
PYTHON_SRC_DIR="/usr/src/Python-${PYTHON_VERSION}"

# 🌐 Download and extract Python source code
cd /usr/src
echo "Downloading Python $PYTHON_VERSION..."
sudo wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz
sudo tar xzf Python-${PYTHON_VERSION}.tgz
cd Python-${PYTHON_VERSION}

# ⚙️ Configure and build Python
echo "Configuring Python build..."
sudo ./configure --enable-optimizations

echo "Building Python (this may take a while)..."
sudo make -j$(nproc)

# 📥 Install without overwriting system Python
echo "Installing Python $PYTHON_VERSION..."
sudo make altinstall

# 🧪 Verify installation
echo "Python $PYTHON_VERSION successfully installed."
python3.11 --version

# 🛠️ Optional: Create symlinks or alias (not system-wide)
echo ""
echo "To use Python 3.11 by default in your shell, you can run:"
echo "  alias python=python3.11"
echo "  alias pip=pip3.11"
echo "Or add those lines to your ~/.bashrc or ~/.zshrc"

# ✅ Done
echo ""
echo "✅ Python $PYTHON_VERSION installation completed!"
