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

## draw hierarchy

To show hierarchy with box-drawing characters,
e.g. in quantitative result tables:
```
Base Method
  ├─ + Plugin A
  └─ + Plugin B
```

use

```tex
\usepackage{pmboxdraw}

\textSFii   % └
\textSFviii % ├
\textSFx    % ─
```

- [Print box drawing characters with pdfLaTeX](https://tex.stackexchange.com/questions/281368/print-box-drawing-characters-with-pdflatex)
- [pmboxdraw – Poor man’s box drawing characters](https://ctan.org/pkg/pmboxdraw)

## wrap figure/table

Let a figure / table be wrapped around by text instead of spanning the whole row,
use `wrapfigure` and `wraptable` in

```tex
\usepackage{wrapfig}
```

References and examples:

- (wrapfigure) [4]
- (wraptable) [Wrap text around a tabular](https://tex.stackexchange.com/questions/49300/wrap-text-around-a-tabular)
- [wrapfig – Produces figures which text can flow around](https://ctan.org/pkg/wrapfig)

## album layout

Album layout: a big figure on one side, several small figures forming a grid on the other side:

```
 _____    ___   ___
|     |  |___| |___|
|     |   ___   ___
|_____|  |___| |___|
```

I use the solution in [1] using the `subcaption` package and `\newbox`, `\bigpicturebox`, `\sbox` and `\usebox` commands.

1. [Big picture with several smaller ones on the side](https://tex.stackexchange.com/questions/302121/big-picture-with-several-smaller-ones-on-the-side)
2. [How can I create this layout of 3 subfigures?](https://tex.stackexchange.com/questions/646910/how-can-i-create-this-layout-of-3-subfigures)
3. [How to place two figures in one column, and then another figure in the 2nd column with double the size?](https://tex.stackexchange.com/questions/611153/how-to-place-two-figures-in-one-column-and-then-another-figure-in-the-2nd-colum)
4. [用floatrow宏包实现插图的异形布置](https://ask.latexstudio.net/ask/article/94.html)
5. [Specific image layout](https://tex.stackexchange.com/questions/49764/specific-image-layout)

# Abbreviations

- Note: (some of) these may be already defined by the template, e.g. the `cvpr.sty` in [4].

```tex
\usepackage{xspace}

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

## encoding

Sometimes there may be some accent character in author names in the bibliography,
e.g.:

- `Stanis{\l}aw Jastrz{\k{e}}bski`

Include the following two packages to avoid compiling errors with `pdflatex`:

```tex
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
```

Reference:

- [Using \c{e} and other special characters in .bib files with pdflatex+biblatex+biber: How to avoid Package inputenc Error?](https://tex.stackexchange.com/questions/311776/using-ce-and-other-special-characters-in-bib-files-with-pdflatexbiblatexbi)

## special symbols

Use [pifont](https://ctan.org/pkg/pifont) for special symbols.
For example:

```tex
\usepackage{pifont}

\ding{51} % ✓
\ding{55} % ✗
```

Also see many symbol and formatting tricks in the TeX source of [Point Transformer V3: Simpler, Faster, Stronger](https://arxiv.org/abs/2312.10035).

# Custom Colours

You can use these colours to configure the citation, link and url colour of `hyperref`.
See [8] for some named colours.

```tex
\usepackage{xcolor} % \textcolor, \definecolor

\definecolor{bilibili}{RGB}{251, 114, 153}
\definecolor{phub}{RGB}{255, 163, 26}
\definecolor{citegreen}{RGB}{34, 139, 34} % FROM: Mask R-CNN (https://arxiv.org/abs/1703.06870)
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
`cleveref` configurations:
```tex
\usepackage[capitalize]{cleveref}

\Crefname{section}{Section}{Sections}
\Crefname{table}{Table}{Tables}
\Crefname{figure}{Figure}{Figures}

\crefname{section}{Sec.}{Secs.}
\crefname{table}{Tab.}{Tabs.}
\crefname{figure}{Fig.}{Figs.}
```

## Detailed Citation

To cite a specific section/figure/theorem/etc. of another paper,
e.g. `[7, Section 2]`,
cite with argument like `\cite[Section 2]{cvpr23abc}`.

# Table of Content

A **local** table of content for the appendix is used in [Point Transformer V3: Simpler, Faster, Stronger](https://arxiv.org/abs/2312.10035).
To do so,
use [etoc](https://ctan.org/pkg/etoc):

```tex
\documentclass{article}
\usepackage{etoc}
\begin{document}

\appendix
\section*{Appendix}

% enable LOCAL ToC for the appendix here
\etoctoccontentsline{part}{Appendix}
\localtableofcontents

\section{Appendix 1}
blabla

\section{Appendix 2}
foo bar

\bibliographystyle{plain}
\bibliography{myBibFile}
\end{document}
```

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
