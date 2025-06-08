Some useful command-line tools.

# fzf

fzf [1] is an interative, general-purpose fuzzy finder.
Fuzzy means you can search strings without exact typing,
e.g. search `a_very_long_var_name` by only typing `avlvn`.

To search some strings in a file with fuzzy matching:
```shell
# on linux
cat FILE | fzf
```
or
```bat
@REM on windows
type FILE | fzf
```

To search only files (ignore folders) under *folder1/folder2/*:
```shell
fzf --walker=file --walker-root=folder1/folder2
# or
cd folder1/folder2
fzf --walker=file
```
To preview file:
```bat
@REM `type` is the `cat` command on Windows.
fzf --preview "type {}"
```
To open the selected file with Vim when pressing Enter within fzf:
```shell
fzf --bind "enter:execute(vim {})"
```

On Linux,
you can use `alias` in .bashrc to set a abbreviation of fzf with these parameters.
On Windows,
you can make a folder under your home directory (e.g. *%USERPROFILE%\scripts/*),
add that to PATH,
and create batch file as alias.
See [fo.bat](scripts/fo.bat) for an example.

# bat

bat [2] is a Linux `cat` with syntax highlighting.
Usage: `bat FILE`.

# delta

delta [3] provides syntax highlighting for git, diff and grep.
Its coniguration is written in *~/.gitconfig*.
See [git/README.md](git/README.md) for an example.

To compare two files,
run `delta FILE_1 FILE_2`.

# Universal ctags

ctags [4] generates tags (anchor points) for files in a code project for code navigation in Vim.
To use it,
you first run `ctags -R .` within a project folder,
which ends up with a file named *tags* within the project.
Then you open a file in the project with vim.
You can use `:ts` or `:tag` and `<Tab>` to see all available tags within the project,
or `:ts /<str>` to search tags with regular expression.

Related vimrc configurations:
```vim
" Automatically look for tags file up the directory tree
set tags=./tags,tags;$HOME

" Case-sensitive tag matching
set tagcase=match

" Show tag matches in preview window
set previewheight=5
nnoremap <C-w>} :ptag <C-r><C-w><CR>

nnoremap <leader>] :ts <C-r><C-w><CR>   " Show tag list for word under cursor
nnoremap <leader>[ :pts <C-r><C-w><CR>  " Preview tag list

" Tag stack navigation
nnoremap <leader>t :tags<CR>            " Show tag stack
nnoremap <leader>n :tn<CR>              " Next tag
nnoremap <leader>p :tp<CR>              " Previous tag
```

# ag

ag [5] is high-speed `grep`.
It searchs a pattern recursively within a path.
For example,
to search lines containing `torch` in a *test.py*: `ag torch test.py`.
Or search `foo` in all files under *bar/*, ignoring case: `ag -i foo bar/`.

# ripgrep

ripgrep [6] is a faster ag?
It respect gitignore rules and skip hidden files/dictories and binary files.
Usage (just replace `ag` with `rg`):
```shell
rg torch test.py
rg -i foo bar/
```

# References

1. [junegunn/fzf](https://github.com/junegunn/fzf)
2. [sharkdp/bat](https://github.com/sharkdp/bat)
3. [dandavison/delta](https://github.com/dandavison/delta)
4. [universal-ctags/ctags](https://github.com/universal-ctags/ctags)
5. [ggreer/the_silver_searcher](https://github.com/ggreer/the_silver_searcher)
6. [BurntSushi/ripgrep](https://github.com/BurntSushi/ripgrep)
