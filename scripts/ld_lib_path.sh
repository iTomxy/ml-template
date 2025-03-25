# Add the lib/ path of current virtual env to LD_LIBRARY_PATH
# to solve the ImportError related to C++ looks like:
#		ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.26' not found (required by ...)
## Usage
# In a running shell script,
# add `. ld_lib_path.sh` before running the training code.
## Example
#   ```shell
#   . ld_lib_path.sh    # dot (.) executing
#   python main.py      # run training code
#   ```

CONDA_P=${1-$HOME/miniconda3}

lib_p=$CONDA_P/envs/$CONDA_DEFAULT_ENV/lib

if [[ "$LD_LIBRARY_PATH" != *"$lib_p"* ]]; then
	if [ -z $LD_LIBRARY_PATH ]; then
		export LD_LIBRARY_PATH=$lib_p
	else
		export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$lib_p
	fi
fi
