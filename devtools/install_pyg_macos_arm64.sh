# A script to install Pytorch geometric on Mchip MacOS
# Author: Xiaorui Dong
# Inspired by this https://medium.com/@jgbrasier/installing-pytorch-geometric-on-mac-m1-with-accelerated-gpu-support-2e7118535c50

CONDA_ENV_NAME="GeoMol"
PYTHON_VERSION="3.12"
SCRIPT_DIR=$(dirname $0)  # Assume the other scripts are available in the same directory as this file

# Function to display usage
usage() {
    echo "Usage: $0 [-n <name>] [--name <name>] [-v <version>] [--version <version>]"
    exit 1
}

# Parse short options (-n and -v)
while getopts ":n:v:" opt; do
    case ${opt} in
        n )
            CONDA_ENV_NAME=$OPTARG
            ;;
        v )
            PYTHON_VERSION=$OPTARG
            ;;
        \? )
            usage
            ;;
    esac
done

# Remove the processed options from the parameters
shift $((OPTIND -1))

# Parse long options (--name and --version)
for arg in "$@"; do
    case $arg in
        --name=*)
            CONDA_ENV_NAME="${arg#*=}"
            shift # Remove --name from processing
            ;;
        --version=*)
            PYTHON_VERSION="${arg#*=}"
            shift # Remove --version from processing
            ;;
        *)
            # Handle unrecognized options
            usage
            ;;
    esac
done

source $SCRIPT_DIR/initialize_conda.sh

if conda env list | grep -qw $CONDA_ENV_NAME; then
    $conda_bin activate $CONDA_ENV_NAME
else
    $conda_bin create -n $CONDA_ENV_NAME python=$PYTHON_VERSION -y
    $conda_bin activate $CONDA_ENV_NAME
fi

PYTHON_VERSION=$(python --version)
echo "Using Python version: $PYTHON_VERSION"

# make sure compiler are correctly installed
$conda_bin install -y clang_osx-arm64 clangxx_osx-arm64 gfortran_osx-arm64

os_version=$(sw_vers -productVersion)

# install PyTorch and pytorch_geometric with the correct compiler
echo "Installing PyTorch..."
MACOSX_DEPLOYMENT_TARGET=$os_version CC=clang CXX=clang++ python -m pip --no-cache-dir install torch torchvision
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")

MACOSX_DEPLOYMENT_TARGET=$os_version CC=clang CXX=clang++ \
python -m pip --no-cache-dir install torch_scatter torch_sparse torch_cluster torch_spline_conv \
-f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cpu.html

MACOSX_DEPLOYMENT_TARGET=$os_version CC=clang CXX=clang++ \
python -m pip --no-cache-dir install torch-geometric

# install other packages
$conda_bin env update -f $SCRIPT_DIR/environment.yml -n $CONDA_ENV_NAME
$conda_bin install nomkl
