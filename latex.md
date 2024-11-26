# Mathematics

```tex
\usepackage{amsmath,amssymb,amsfonts,amsthm}
% \usepackage{bm} % defines \bm
\usepackage{mathtools} % hollow char, see [1]
\usepackage{stmaryrd} % hollow bracket, see [1]

% argmax, argmin
\DeclareMathOperator*{\argmin}{\arg\min}
\DeclareMathOperator*{\argmax}{\arg\max}
% iverson bracket
\DeclarePairedDelimiter{\iverson}{\llbracket}{\rrbracket}
```

# Algorithm/Pseudo-code & Code

```tex
% pseudo code (algorithm)
\usepackage{algorithm}
\usepackage{algorithmic}

% code
\usepackage{listings}
```

# Figure & Table

```tex
% table
\usepackage{bigstrut}
\usepackage{booktabs}
\usepackage{caption}
\usepackage{extarrows}
\usepackage{makecell}
\usepackage{multirow}

% figure
\usepackage{graphicx}
%\usepackage{subfigure}
\usepackage{subcaption} % use it to include subfigure
```

# Abbreviations

```tex
\usepackage{xspace}

% abbreviations
\makeatletter
\DeclareRobustCommand\onedot{\futurelet\@let@token\@onedot}
\def\@onedot{\ifx\@let@token.\else.\null\fi\xspace}
\def\eg{\emph{e.g}\onedot} \def\Eg{\emph{E.g}\onedot}
\def\ie{\emph{i.e}\onedot} \def\Ie{\emph{I.e}\onedot}
\def\viz{\emph{viz}\onedot}
\def\cf{\emph{cf}\onedot} \def\Cf{\emph{Cf}\onedot}
\def\etc{\emph{etc}\onedot}
\def\vs{\emph{vs}\onedot} % `vs.` not `v.s.`
\def\wrt{w.r.t\onedot} \def\dof{d.o.f\onedot}
\def\iid{i.i.d\onedot} \def\wolog{w.l.o.g\onedot}
\def\etal{\emph{et al}\onedot}
\makeatother
```

# Text & Symbol

```tex
\usepackage{soul} % \st: stridethrough text

% \bs := \boldsymbol
\ifdefined\bm
  \PackageWarning{\jobname}{\string\bm\space is already defined.}
\else
  \newcommand{\bm}{\boldsymbol}
\fi
```

# Custom Colours

You can use these colours to configure the citation, link and url colour of `hyperref`.
See [8] for some named colours.

```tex
\usepackage{xcolor} % \textcolor, \definecolor

\definecolor{bilibili}{RGB}{251, 114, 153}
\definecolor{phub}{RGB}{255, 163, 26}
\definecolor{halfgreen}{RGB}{0, 153, 0} % FROM: https://arxiv.org/abs/1805.08193
\definecolor{kaiming-green}{RGB}{57, 181, 74} % FROM: https://arxiv.org/abs/2402.10739
\definecolor{royalblue}{RGB}{0, 113, 188}
\definecolor{halfblue}{RGB}{0, 0, 128} % ICML citation blue
```

# Citation & Reference

```tex
% url & cite
\usepackage[
  pagebackref,
  breaklinks,
  colorlinks,
  bookmarks=true,
  citecolor=royalblue % custom colour defined above
]{hyperref}

% easy reference of figure, table, section
% - `\eqref`: built in the `amsmath` package
% - `\nameref`: built in the `nameref` package
% - `\pageref`: built-in
\providecommand{\eqnref}[1]{Eq.~\eqref{#1}}
\providecommand{\Eqnref}[1]{Equation~\eqref{#1}}
\providecommand{\figref}[1]{\figurename~\ref{#1}}
\providecommand{\tabref}[1]{\tablename~\ref{#1}}
\providecommand{\secref}[1]{\S\ref{#1}}
\providecommand{\Secref}[1]{Section~\ref{#1}}
```

~~Alternative: use `\autoref` provided in `hyperref` package~~.

Alternative: use [cleveref](https://au.mirrors.cicku.me/ctan/macros/latex/contrib/cleveref/cleveref.pdf).

## Detailed Citation

To cite a specific section/figure/theorem/etc. of another paper,
e.g. `[7, Section 2]`,
cite with argument like `\cite[Section 2]{cvpr23abc}`.

# Quotation Marks

```tex
"a", ``b'', 'c', `d'.
```

# Fonts

- [LaTeX/Fonts](https://en.wikibooks.org/wiki/LaTeX/Fonts)
- [What are all the font styles I can use in math mode?](https://tex.stackexchange.com/questions/58098/what-are-all-the-font-styles-i-can-use-in-math-mode)
- [Superscript outside math mode](https://tex.stackexchange.com/questions/47324/superscript-outside-math-mode)

# Figure in Title & Text

See TeX source of:

- (arxiv'24) PointMamba: A Simple State Space Model for Point Cloud Analysis - [arXiv](https://arxiv.org/abs/2402.10739)

Its custom `\charimage` command:

```tex
\newcommand{\charimage}[1]{\raisebox{-0.25\height}{\includegraphics[height=\baselineskip]{figure/logo.png}}}
```

Its hacked `\title`:
```tex
\title{
\vspace{-1.em}
\renewcommand{\windowpagestuff}{
\quad\includegraphics[width=1.cm, trim={0cm 10cm 0cm 0cm}, clip=False]{figure/logo.png}
}
\begin{cutout}{0}{0.3cm}{22cm}{1}
\vspace{-30pt}
{\color{white} empty} \protect\linebreak
\protect\linebreak
{\color{white} \qquad } PointMamba: A Simple State Space Model for Point Cloud Analysis
\end{cutout}
\vspace{-0.7em}
}
```

# References

1. [latex空心小写字母、数字、括号](https://blog.csdn.net/HackerTom/article/details/134221777)
2. [latex自定义缩写](https://blog.csdn.net/HackerTom/article/details/134233080)
3. [latex cite命令、款式](https://blog.csdn.net/HackerTom/article/details/134318666)
4. [guanyingc/cv_rebuttal_template](https://github.com/guanyingc/cv_rebuttal_template)
5. [guanyingc/latex_paper_writing_tips](https://github.com/guanyingc/latex_paper_writing_tips)
6. [MLNLP-World/Paper-Writing-Tips](https://github.com/MLNLP-World/Paper-Writing-Tips)
7. [MLNLP-World/Paper-Picture-Writing-Code](https://github.com/MLNLP-World/Paper-Picture-Writing-Code)
8. [LaTeX Color](https://latexcolor.com/)
