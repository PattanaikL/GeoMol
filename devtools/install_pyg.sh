# A script to install Pytorch geometric on normal platform
# Author: Xiaorui Dong
# Inspired by this https://medium.com/@jgbrasier/installing-pytorch-geometric-on-mac-m1-with-accelerated-gpu-support-2e7118535c50

CONDA_ENV_NAME="GeoMol"
PYTHON_VERSION="3.12"
CUDA_VERSION="cpu"
SCRIPT_DIR=$(dirname $0)  # Assume the other scripts are available in the same directory as this file

# Function to display usage
usage() {
    echo "Usage: $0 [-n <environment name>] [--name <environment name>] [-v <version>] [--python-version <version>] [- <cuda version>] [--cuda-version <cuda version>]"
    exit 1
}

# Parse short options (-n and -v)
while getopts ":n:v:c:" opt; do
    case ${opt} in
        n )
            CONDA_ENV_NAME=$OPTARG
            ;;
        v )
            PYTHON_VERSION=$OPTARG
            ;;
        c )
            CUDA_VERSION=$OPTARG
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
        --python_version=*)
            PYTHON_VERSION="${arg#*=}"
            shift # Remove --version from processing
            ;;
        --cuda_version=*)
            CUDA_VERSION="${arg#*=}"
            shift # Remove --version from processing
            ;;
        *)
            usage
            ;;
    esac
done

# parse cuda
# Using regex to capture the major and minor version numbers for detailed matching
if [[ "$(uname)" != 'Darwin' ]]; then
    if [[ $CUDA_VERSION =~ ^([0-9]+)\.([0-9]+)(\.([0-9]+))?$ ]]; then
        major_version="${BASH_REMATCH[1]}"
        minor_version="${BASH_REMATCH[2]}"
        cuda_version_formatted="${major_version}.${minor_version}"

        # Construct the CUDA and CUDA_VERSION variables based on input
        CUDA="cudatoolkit=$cuda_version_formatted"
        CUDA_VERSION="cu${major_version}${minor_version}"
    elif [ "$cuda_input" == "cpu" ]; then
        # For CPU-only selection
        CUDA="cpuonly"
        CUDA_VERSION="cpu"
    else
        echo "Invalid input. Please ensure you enter a valid CUDA version in the format xx.xx or 'cpu'."
        exit 1
    fi
else
    CUDA="cpuonly"
    CUDA_VERSION="cpu"
fi
echo "You selected CUDA version: $CUDA_VERSION ($CUDA)"

source $SCRIPT_DIR/initialize_conda.sh

if conda env list | grep -qw $CONDA_ENV_NAME; then
    $conda_bin activate $CONDA_ENV_NAME
else
    $conda_bin create -n $CONDA_ENV_NAME python=$PYTHON_VERSION -y
    $conda_bin activate $CONDA_ENV_NAME
fi

# check Python version
PYTHON_VERSION=$(python --version)
echo "Using Python version: $PYTHON_VERSION"

# install PyTorch
echo "Installing PyTorch with requested CUDA version $CUDA_VERSION..."
# echo "Running: conda install pytorch torchvision $CUDA -c pytorch -y"
# $conda_bin install pytorch torchvision $CUDA -c pytorch -y
echo "Running: pip install torch torchvision"
pip install torch torchvision --index-url https://download.pytorch.org/whl/$CUDA_VERSION

# get PyTorch version
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
if [ -n $TORCH_VERSION ]; then
    echo "Using PyTorch version: $TORCH_VERSION"
else
    echo "Cannot find a matched PyTorch version with $CUDA_VERSION for Python $PYTHON_VERSION. Exit."
    # echo "Removing the installed environment"
    # source deactivate
    # $conda_bin env remove -n $environmentName
    exit 1
fi

# install torch_geometric
echo "Installing torch-geometric..."
echo "Using CUDA version: $CUDA_VERSION"
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
pip install torch-geometric

# install other package
$conda_bin env update -f $SCRIPT_DIR/environment.yml -n $CONDA_ENV_NAME