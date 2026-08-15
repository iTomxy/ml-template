Utility scripts.

- [bright.bat](bright.bat) & [ps_brightness.ps1](ps_brightness.ps1): set system brightness in command line
- [dfs.sh](dfs.sh): do something recursively under a folder
- [find-gpu.sh](find-gpu.sh): filter GPUs, and set CUDA_VISIBLE_DEVICES
- [fkill.sh](fkill.sh)

# Compare Two Files

Check whether two files (FILE1 vs. FILE2) are identical on Windows.

## certutil

1. Compute hash: `certutil -hashfile <FILE> SHA256`
2. Compare two hash manually.

## fc /b

```
fc /b FILE1 FILE2
```
If identical, you shall see:
```
FC: no differences encountered
```

## powershell

```
(Get-FileHash FILE1).Hash -eq (Get-FileHash FILE2).Hash
```
You shall see `False` if they differ.
