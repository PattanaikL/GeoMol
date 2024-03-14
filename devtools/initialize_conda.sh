echo "Initializing Conda..."

if which mamba > /dev/null; then
    conda_bin="mamba"
    echo "As mamba is available, using mamba by default..."
else
    conda_bin="conda"
fi

conda_base_dir=$(dirname $(dirname $CONDA_EXE))

if [ "$conda_bin" = "mamba" ]; then
    source "$conda_base_dir/etc/profile.d/conda.sh"
    source "$conda_base_dir/etc/profile.d/mamba.sh"
else
    source "$conda_base_dir/etc/profile.d/conda.sh"
fi

export conda_bin
