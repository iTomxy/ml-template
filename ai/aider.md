User aider [1] with openrouter [2] API key to use free models.

1. export `OPENROUTER_API_KEY` in ~/.bashrc
2. check available free models with: `aider --list-models openrouter/ | grep free`.
But not all the listed models are currently available for free (obsolete?).
3. compare and select a model according to benchmarks, e.g. [3].
4. configure ~/.bashrc for eacy using:
    ```bash
    export OPENROUTER_API_KEY="sk-"
    aider_model="openrouter/openrouter/free" # substitute with the picked model
    # Use `whereis aider` to locate
    aider() { $HOME/.local/bin/aider --model $aider_model --yes --edit-format whole "$@"; }
    ```

# References

1. [aider](https://aider.chat/): [github](https://github.com/Aider-AI/aider)
2. [openrouter](https://openrouter.ai/)
3. [Artificial Analysis Intelligence Index](https://artificialanalysis.ai/evaluations/artificial-analysis-intelligence-index)
