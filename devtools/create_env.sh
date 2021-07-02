# Developed by Kevin A. Spiekermann
# This script does the following tasks:
# 	- creates the conda
# 	- prompts user for desired CUDA version
# 	- installs PyTorch with specified CUDA version in the environment
# 	- installs torch torch-geometric in the environment


# get OS type
unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     machine=Linux;;
    Darwin*)    machine=MacOS;;
    CYGWIN*)    machine=Cygwin;;
    MINGW*)     machine=MinGw;;
    *)          machine="UNKNOWN:${unameOut}"
esac
echo "Running ${machine}..."


# request user to select one of the supported CUDA versions
# source: https://pytorch.org/get-started/locally/
PS3='Please enter 1, 2, 3, or 4 to specify the desired CUDA version from the options above: '
options=("9.2" "10.1" "10.2" "cpu" "Quit")
select opt in "${options[@]}"
do
    case $opt in
        "9.2")
            CUDA="cudatoolkit=9.2"
            CUDA_VERSION="cu92"
            break
            ;;
        "10.1")
			CUDA="cudatoolkit=10.1"
            CUDA_VERSION="cu101"
            break
            ;;
        "10.2")
			CUDA="cudatoolkit=10.2"
            CUDA_VERSION="cu102"
            break
            ;;
        "cpu")
			# "cpuonly" works for Linux and Windows
			CUDA="cpuonly"
			# Mac does not use "cpuonly"
			if [ $machine == "Mac" ]
			then
				CUDA=" "
			fi
            CUDA_VERSION="cpu"
            break
            ;;
        "Quit")
            exit
            ;;
        *) echo "invalid option $REPLY";;
    esac
done

echo "Creating conda environment..."
echo "Running: conda env create -f environment.yml"
conda env create -f devtools/environment.yml

# activate the environment to install torch-geometric
source activate GeoMol

echo "Installing PyTorch with requested CUDA version..."
echo "Running: conda install pytorch torchvision $CUDA -c pytorch"
conda install pytorch torchvision $CUDA -c pytorch

echo "Installing torch-geometric..."
echo "Using CUDA version: $CUDA_VERSION"
# get PyTorch version
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
echo "Using PyTorch version: $TORCH_VERSION"

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
pip install torch-geometric
