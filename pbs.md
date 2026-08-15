UTS launches the Cetus cluster on 2026,
which is an alternative of the old iHPC cluster
and is PBS-based.

- account request: [eResearch Compute : CETUS Account Request, Private Node and AWS](https://utsprodesm.service-now.com/serviceconnect?id=sc_cat_item&sys_id=3487f42c83e9e690397ac160ceaad3f5&table=sc_cat_item) <- from e-mail
- document: [1]
- login IP: `hpc-login01.hpc.uts.edu.au`
- login account: `u` + <UTS 8-digit ID>, e.g. `u12345678`.

# Interative Environment

Connect to a compute node with an interative environment using:
```
qsub -I -q large_gpuq -l select=1:ngpus=1:ncpus=2:mem=15gb -l walltime=00:30:00
```
where:
- `-I`: for interative
- `-q`: which queue. Use `qstat -Q` to see what queues are available.
- `-l`: to reques resources
    - `select`: how many nodes
    - `walltime`: max requested running time, after which the system may
    kill the session.
    - `host`: which node to connect to. Use `pbsnodes -a` to see available nodes.

See [2] for more.

# Inspection

On login node:

- `qstat -Q`: what queues are available. See [3].
- `pbsnodes -a`: show all nodes and available resources. See [4].

On a compute node:

- `module avail`: what modules are available to be loaded, e.g. cuda.

# Example

PBS script is a shell script with PBS commands starting with `#PBS`.
Those commands are comments to the shell.

```shell
# train_mnist.sh
#PBS -N train-mnist
#PBS -o pbso.train-mnist
#PBS -e pbse.train-mnist
#PBS -q large_gpuq
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb
#PBS -l walltime=01:00:00


# 1. load needed modules
module load cuda/12.8

# 2. go to workspace
cd $HOME/codes/mnist

# 3. activate python environment
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate pt251_cu128

# 4. run
python train_mnist.py
```

where:
- `-N`: job name

- `-o`, `-e`: path to the file for stdout & stderr.
If ending with `/`, then they specify the folder, not file name.
If not specified, the output files will be `<JOB_NAME>.o<JOB_ID>` and `<JOB_NAME>.e<JOB_ID>`.
Use `#PBS -j oe` to join stderr into stdout,
so that only one stdout file `<JOB_NAME>.o<JOB_ID>` is created.

- `-q`: select a queue

- `-l`: require resources.
Both using multiple `-l` commands or squeezing them into one with `:` separation are accepted.

Then submit this job with:
```shell
cd $HOME/codes/mnist
qsub train_mnist.sh
```

# Pass Arguments at `qsub`

When submitting to PBS a script that accepts arguments,
pass the arguments with environment variables.
E.g.:

```shell
# test.sh
#PBS -N test
#PBS -q large_gpuq
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb
#PBS -l walltime=01:00:00

# Accept arguments via environment varialbes `PRED_PATH` AND `GT_PATH`.
pred_path=${PRED_PATH:-./log/pred}
gt_path=${GT_PATH:-./data/gt}

python test.py --pred-path $pred_path --gt-path $gt_path
```

and specify those environment with the `-v` flag,
separated with `,`:

```shell
qsub -v "PRED_PATH=./log/exp1/pred", GT_PATH=./data/data2/gt test.sh
```

# References

1. [eResearch HPC Documentation](https://hpc.research.uts.edu.au/getting_started/)
2. [Accessing GPU Nodes](https://hpc.research.uts.edu.au/gpu/gpu/)
3. [Getting Information on the Queues](https://hpc.research.uts.edu.au/pbs/queues/)
4. [Getting Information on the Nodes](https://hpc.research.uts.edu.au/pbs/nodes/)
