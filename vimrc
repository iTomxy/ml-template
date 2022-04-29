let g:iswindows = 0
let g:islinux = 0
if (has("win32") || has("win64") || has("win95") || has("win16"))
	let g:iswindows = 1
else
	let g:islinux = 1
endif


"--- Vundle ---"
set nocompatible
filetype off
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin('~/vim-files/vim-plugin')
Plugin 'VundleVim/Vundle.vim'
Plugin 'tpope/vim-commentary'
Plugin 'preservim/nerdtree'
Plugin 'Vimjas/vim-python-pep8-indent'
Plugin 'jiangmiao/auto-pairs'
Plugin 'altercation/solarized'
Plugin 'tomasiser/vim-code-dark'
call vundle#end()


"--- Setting ---"
"set nocompatible
filetype plugin indent on
set autoread
set autochdir
set nobackup
"set noundofile
set undodir=~/vim-files/vim-undo
set hlsearch
set incsearch
set backspace=indent,eol,start
let mapleader = '\'


"--- Encoding ---"
set encoding=utf-8
set fileencoding=utf-8
set termencoding=chinese
if 1 == g:iswindows
	source $VIMRUNTIME/delmenu.vim
	source $VIMRUNTIME/menu.vim
endif
"language messages zh_CN.utf-8
"language messages none
"set langmenu=none
"set fileformats=unix,dos
"set fileformat=unix " use `\n`
"set fileformat=dos " use `\r\n`


"--- Tab ---"
set tabstop=4
set softtabstop=4
set shiftwidth=4
set smartindent
set cindent
set expandtab
autocmd FileType h,c,cpp,java set noexpandtab


"--- Theme ---"
syntax enable
syntax on
set number
set ruler
set showcmd
set showmode
set showmatch
"set ignorecase
set smartcase
set cursorline
if 1 == g:iswindows
	set t_Co=256
	set t_ut=
	if has ("gui_running")
		"set cursorline
		colorscheme solarized
	else
		colorscheme codedark
		"colorscheme desert
	endif
	set guifont=Consolas:h13:cANSI
	set guifontwide=YouYuan:h13
endif
"set listchars=precedes:<,extends:>,tab:\|\ ,eol:¬,space:·
"set list


"--- adjust background according time: windows GUI + solarized ---"
if 1 == g:iswindows
	"autocmd GUIEnter * simalt ~x
	autocmd GUIEnter * call Init_GUI()
endif

func! Init_GUI()
    " config window
    set guioptions-=m
    set guioptions-=T
    " simalt ~x
    set lines=32 columns=110
    " choose background
    let night_hour = 17
    let night_minute = 30
    let hour = strftime('%H')
    let minute = strftime('%M')
    " echo hour . ":" . minute
    if hour < night_hour
        set background=light
    elseif (hour == night_hour) && (minute < night_minute)
        set background=light
    else
        set background=dark
    endif
endfunc


"--- Buffer ---"
" forward
noremap <C-Left> gt
inoremap <C-Left> <ESC>gta
vnoremap <C-Left> <ESC>gt
" backward
noremap <C-Right> gT
inoremap <C-Right> <ESC>gTa
vnoremap <C-Right> <ESC>gT
" open tab
noremap <C-t> :tabe 
inoremap <C-t> <ESC>:tabe 
vnoremap <C-t> <ESC>:tabe 
" close tab <- `Ctrl-W` conflicts with window switching
"noremap <C-w> :q<CR>
"inoremap <C-w> <ESC>:q<CR>
"vnoremap <C-w> <ESC>:q<CR>


"--- NERDTree ---"
" F2 toggle open/close
noremap <F2> :NERDTreeToggle<CR>
inoremap <F2> <ESC>:NERDTreeToggle<CR>
" Ctrl + F2 specify path & open
noremap <C-F2> :NERDTree
inoremap <C-F2> <ESC>:NERDTree


"--- Code Folding ---"
set foldenable
set foldmethod=manual
" Ctrl-f toggle open/zip
nnoremap <C-f> @=((foldclosed(line('.')) < 0) ? 'zc':'zo')<CR>


"--- select all ---"
inoremap <C-a> <ESC>ggvG$
noremap <C-a> ggvG$


"--- switch windows ---"
nmap <silent> <leader>mw :call MarkWindowSwap()<CR>
nmap <silent> <leader>pw :call DoWindowSwap()<CR>

function! MarkWindowSwap()
    let g:markedWinNum = winnr()
endfunction

function! DoWindowSwap()
    "Mark destination
    let curNum = winnr()
    let curBuf = bufnr( "%" )
    exe g:markedWinNum . "wincmd w"
    "Switch to source and shuffle dest->source
    let markedBuf = bufnr( "%" )
    "Hide and open so that we aren't prompted and keep history
    exe 'hide buf' curBuf
    "Switch to dest and shuffle source->dest
    exe curNum . "wincmd w"
    "Hide and open so that we aren't prompted and keep history
    exe 'hide buf' markedBuf 
endfunction


"--- F9 compile & run ---"
noremap <F9> :call Compile_iTom()<CR>
inoremap <F9> <ESC>:call Compile_iTom()<CR>

func! Compile_iTom()
	exec "w"
	if &filetype == 'c'
		exec "! gcc % -o %<"
		exec "! ./%<"
	elseif &filetype == 'cpp'
		exec "! g++ % -o %<"
		exec "! ./%<"
	elseif &filetype == 'java'
		exec "! javac %"
		exec "! java %<"
	elseif &filetype == 'python'
		exec "! python %"
	elseif &filetype == 'matlab'
		exec "! octave %"
	endif
endfunc


"--- < & > ---"
"autocmd FileType xml,html inoremap < <lt>><Left>|
"	inoremap > <C-r>=Close_Pair('>')<CR>


"--- open pair ---"
"inoremap ( <C-r>=Open_Pair('(', ')')<CR>
"inoremap [ <C-r>=Open_Pair('[', ']')<CR>
"inoremap { <C-r>=Open_Pair('{', '}')<CR>

func! Open_Pair(open, close)
    let line = getline('.')
	let nxt_char = line[col('.') - 1]
    if col('.') > strlen(line) || nxt_char == ' '
		return a:open.a:close."\<Left>"
    elseif nxt_char == a:close
        return a:open
	elseif nxt_char == ')' || nxt_char == ']' || nxt_char == '}'
		return a:open.a:close."\<Left>"
	else
		return a:open
    endif
endfunc


"--- close pair ---"
"inoremap ) <C-r>=Close_Pair(')')<CR>
"inoremap ] <C-r>=Close_Pair(']')<CR>
"inoremap } <C-r>=Close_Pair('}')<CR>

func! Close_Pair(char)
	if getline('.')[col('.') - 1] == a:char
		return "\<Right>"
	else
		return a:char
	endif
endfunc


"--- quote ---"
"inoremap " <C-r>=Same_Pair('"')<CR>
"inoremap ' <C-r>=Same_Pair("'")<CR>
"inoremap ` <C-r>=Same_Pair('`')<CR>

func! Same_Pair(char)
    let line = getline('.')
	let pair = a:char.a:char."\<Left>"
    if col('.') > strlen(line) " || line[col('.') - 1] == ' '
        return pair
    elseif line[col('.') - 1] == a:char
        return "\<Right>"
    else
		let pre_char = line[col('.') - 2]
		let nxt_char = line[col('.') - 1]
		if pre_char == '(' && nxt_char == ')'
			return pair
        elseif pre_char == '[' && nxt_char == ']'
            return pair
        elseif pre_char == '{' && nxt_char == '}'
            return pair
        elseif pre_char == '<' && nxt_char == '>'
            return pair
		else
			return a:char
		endif
    endif
endfunc


"--- carrage return ---"
"autocmd FileType h,c,cpp,java,python inoremap <CR> <C-r>=Carrage_Return()<CR>

func! Carrage_Return()
	let line = getline('.')
	if line[col('.') - 2] == '{' && line[col('.') - 1] == '}'
		return "\<CR>\<Up>\<ESC>A\<CR>"
	else
		return "\<CR>"
    endif
endfunc


"--- backspace ---"
"inoremap <BS> <C-r>=Back_Space()<CR>

func! Back_Space()
	let line = getline('.')
	let pre_char = line[col('.') - 2]
	let nxt_char = line[col('.') - 1]
	let del_pair = "\<BS>\<DEL>"
	if pre_char == '(' && nxt_char == ')'
		return del_pair
	elseif pre_char == '[' && nxt_char == ']'
		return del_pair
	elseif pre_char == '{' && nxt_char == '}'
		return del_pair
	elseif pre_char == '<' && nxt_char == '>'
		return del_pair
	elseif pre_char == '"' && nxt_char == '"'
		return del_pair
	elseif pre_char == "'" && nxt_char == "'"
		return del_pair
	elseif pre_char == '`' && nxt_char == '`'
		return del_pair
	else
		return "\<BS>"
	endif
endfunc


"--- tabulator ---"
"autocmd FileType h,c,cpp,java,python inoremap <S-TAB> <C-r>=Tabulator()<CR>

func! Tabulator()
    if strlen(getline('.')) < 1
        return "\<BS>\<CR>"
    else
        return "\<TAB>"
    endif
endfunc


"--- before saving ---"
autocmd BufWritePre * call Before_Saving()

func! Before_Saving()
    " delete trailing white space
    exec "%s/\s\+$//e"
    " delete trailing blank lines
    let ln_nb = prevnonblank('$')
    let ln_eof = line('$')
    if ln_eof > ln_nb + 1
        exec (ln_nb + 1) . "," . ln_eof . "d"
    endif
    " add 1 trailing blank line
    if prevnonblank('$') == line('$')
        call append(line('$'), "")
    endif
endfunc
