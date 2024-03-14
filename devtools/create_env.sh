# Developed by Xiaorui Dong and Kevin A. Spiekermann
# This script does the following tasks:
# 	- creates the conda
# 	- installs PyTorch with specified CUDA version in the environment
# 	- installs torch torch-geometric in the environment

SCRIPT_DIR=$(dirname $0)

CONDA_ENV_NAME="GeoMol"

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

if [ "$machine" != "MacOS" ]; then
    # Prompt the user to input their desired CUDA version or 'cpu'
    echo "Please input your desired CUDA version in the format xx.xx (e.g., 10.2, 12.3) or 'cpu' for no CUDA available:"
    read cuda_input

if [ "$machine" == "MacOS" ] && [ "$(uname -m)" == "arm64" ]; then

    $SHELL $SCRIPT_DIR/install_pyg_macos_arm64.sh -n $CONDA_ENV_NAME

else

    source $SCRIPT_DIR/install_pyg.sh -n $CONDA_ENV_NAME -c $cuda_input

fi
