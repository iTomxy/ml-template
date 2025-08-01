# Project Structure

```
draft/
|- preamble.tex
|- section_1.tex
|- section_n.tex
|- fig/
|  |- *.pdf
|  |- *.png
|- fig-<LABEL>.tex  # per figure
|- tab-<LABEL>.tex  # per table
|- ref-dataset.bib  # separate different kinds of references
|- ref-baseline.bib
|- *.bib
|- main.tex         # entry .tex file that \input others
|- *.bst
|- *.dtx
`- *.sty
```

Use `\graphicspath{{fig/}}` and `\input{}`.

## combining .tex files

Sometimes you want to combine all .tex files in your LaTeX project into one,
e.g. when .tex and .bib files are required to be uploaded to the submission system.
Use the `latexpand` command (comes with a local TeX Live installation) to do this:
- *main.tex*: the entry .tex file of your LaTeX project
- *combined.tex*: the output combined .tex file
```shell
latexpand -o combined.tex --out-encoding 'encoding(UTF-8)' main.tex
```
It deals with embedded `\input` commands automatically.
But:
- always remember to compile the output combined .tex file to verify its integrity before submission.
- .bib files are not combined. Submit them all manually if there are more than one.

# Mathematics

```tex
\usepackage{amsmath,amssymb,amsfonts,amsthm}
% \usepackage{bm} % defines \bm
\usepackage{mathtools} % hollow char, see [1]
\usepackage{stmaryrd} % hollow bracket, see [1]

% argmax, argmin
\DeclareMathOperator*{\argmin}{\arg\min}
\DeclareMathOperator*{\argmax}{\arg\max}
```

## paired delimiters

Combine the opening and closing bracket into a single command to avoid missing
and achieve easy auto-sizing (`\left` and `\right`).

- use `\DeclarePairedDelimiter` to define.
- add `*` to enable auto-sizing. E.g. `\paren{x}` gives `(x)`, while `\paren*{x}` gives `\left( x \right)`.

```tex
\DeclarePairedDelimiter{\iverson}{\llbracket}{\rrbracket} % Iverson bracket
\DeclarePairedDelimiter{\ceil}{\lceil}{\rceil}
\DeclarePairedDelimiter{\floor}{\lfloor}{\rfloor}
\DeclarePairedDelimiter{\paren}{(}{)} % PARENtheses
\DeclarePairedDelimiter{\sqrbrk}{[}{]} % SQuaRe BRacKets
\DeclarePairedDelimiter{\curbrk}{\{}{\}} % CURly BRacKets
\DeclarePairedDelimiter{\card}{|}{|} % CARDinality
\DeclarePairedDelimiter{\norm}{\|}{\|}

% examples
\begin{align}
    \paren{\sum^{A^*_t}_{i=\theta_0} x} \quad
    \paren*{\sum^{A^*_t}_{i=\theta_0} x} \\
    \sqrbrk{\sum^{A^*_t}_{i=\theta_0} x} \quad
    \sqrbrk*{\sum^{A^*_t}_{i=\theta_0} x} \\
    \curbrk{\sum^{A^*_t}_{i=\theta_0} x} \quad
    \curbrk*{\sum^{A^*_t}_{i=\theta_0} x} \\
    \card{\sum^{A^*_t}_{i=\theta_0} x} \quad
    \card*{\sum^{A^*_t}_{i=\theta_0} x} \\
    \norm{\sum^{A^*_t}_{i=\theta_0} x} \quad
    \norm*{\sum^{A^*_t}_{i=\theta_0} x} \\
    \iverson{\sum^{A^*_t}_{i=\theta_0} x} \quad
    \iverson*{\sum^{A^*_t}_{i=\theta_0} x} \\
    \floor{\sum^{A^*_t}_{i=\theta_0} x} \quad
    \floor*{\sum^{A^*_t}_{i=\theta_0} x} \\
    \ceil{\sum^{A^*_t}_{i=\theta_0} x} \quad
    \ceil*{\sum^{A^*_t}_{i=\theta_0} x}
\end{align}
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

## gif

To show a gif in pdf, use [animate](https://ctan.org/pkg/animate). A minimal example adapted from [Magic Mirror: ID-Preserved Video Generation in Video Diffusion Transformers](https://arxiv.org/abs/2501.03931):

```tex
\documentclass{article}
\usepackage{graphicx}
\usepackage{animate}
\begin{document}

%\begin{minipage}[b]{\textwidth}
%\centering
\animategraphics[width=\linewidth,autoplay,loop,every=2]{6}{D:/figs/gifs/output1_frames_blended/}{0}{47}
%\end{minipage}

\end{document}
```

However, pdf opened in browser only shows one static image instead of a gif. Has to see it in Adobe Acrobat.

## colourful table cell

Fill background colour of some columns/rows/cell of a table.
An example:

- `\usepackage[table]{xcolor}`: \columncolor, \rowcolor
- `\usepackage{colortbl}`: \cellcolor
- `\usepackage{pgfplotstable}`: more advanced usages

```tex
% \usepackage[table]{xcolor}
% \usepackage{colortbl}

\begin{table}[h]
    \centering
    \begin{tabular}{c>{\columncolor{blue!20}}c} % column bg colour
    \hline
    Column 1 & Column 2 \\
    \hline
    \cellcolor{citegreen!20}Data 1 & Data 2 \\ % cell bg colour
    \rowcolor{gray!20} % row bg colour
    Data 3 & Data 4 \\
    \hline
    \end{tabular}
    \caption{caption}
\end{table}
```

## table in figure

Inspect the TeX source of figure 1 in [(iclr'25) Hydra-SGG: Hybrid Relation Assignment for One-stage Scene Graph Generation](https://arxiv.org/abs/2409.10262): use `\put` and `\scalebox` to write those texts and table cells manually.

# Abbreviations

- Note: (some of) these may be already defined by the template, e.g. the `cvpr.sty` in [4].

```tex
\usepackage{xspace}

\makeatletter
\DeclareRobustCommand\onedot{\futurelet\@let@token\@onedot}
\def\@onedot{\ifx\@let@token.\else.\null\fi\xspace}
\def\aka{\emph{a.k.a}\onedot}
\def\eg{\emph{e.g}\onedot} \def\Eg{\emph{E.g}\onedot}
\def\ie{\emph{i.e}\onedot} \def\Ie{\emph{I.e}\onedot}
\def\viz{\emph{viz}\onedot}
\def\cf{\emph{cf}\onedot} \def\Cf{\emph{Cf}\onedot}
\def\etc{\emph{etc}\onedot}
\def\vs{\emph{vs}\onedot} % `vs.` not `v.s.`
\def\wrt{w.r.t\onedot}
\def\dof{d.o.f\onedot}
\def\iid{i.i.d\onedot}
\def\wolog{w.l.o.g\onedot}
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

Use [pifont](https://ctan.org/pkg/pifont) and [textcomp](https://docs.mathjax.org/en/latest/input/tex/extensions/textcomp.html) for special symbols.
For example:

```tex
\usepackage{pifont}

\ding{51} % ✓
\ding{55} % ✗
```

and

```tex
\usepackage{textcomp}
\texttimes % ×
\textdiv % ÷
\textpm % ±
\textdegree % °
\textdagger % †
\textdaggerdbl % ‡
\textparagraph % ¶
\textmusicalnote % ♪
```

Also see many symbol and formatting tricks in the TeX source of [Point Transformer V3: Simpler, Faster, Stronger](https://arxiv.org/abs/2312.10035).

# Custom Colours

You can use these colours to configure the citation, link and url colour of `hyperref`.
See [LaTeX Color](https://latexcolor.com/) and [Chinese Colors](https://zhongguose.com/) for some named colours.

```tex
\usepackage[table]{xcolor} % \textcolor, \definecolor

\definecolor{bilibili}{RGB}{251, 114, 153}
\definecolor{phub}{RGB}{255, 163, 26}
\definecolor{citegreen}{RGB}{34, 139, 34} % FROM: Mask R-CNN (https://arxiv.org/abs/1703.06870)
\definecolor{royalblue}{RGB}{0, 113, 188}
\definecolor{halfblue}{RGB}{0, 0, 128} % ICML citation blue
```

## temporarily define colour

```tex
% speficy RGB in-place
\textcolor[RGB]{0, 113, 188}{foo bar}
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

## Cite with Text

To cite a specific section/figure/theorem/etc. of another paper,
e.g. `[7, Section 2]`,
cite with argument like `\cite[Section 2]{cvpr23abc}`.

To add text both before and/or after the citation like `[DDPMs; 7, Section 2]`
(seen in [Diffusion models as plug-and-play priors](https://arxiv.org/abs/2206.09012)),
you should use `\usepackage{natbib}`:
```tex
% \usepackage{natbib}
\cite[DDPMs;][Section 2]{cvpr23abc}
```
Refer to [Natbib citation styles](https://www.overleaf.com/learn/latex/Natbib_citation_styles).

# Numbering of Table/Figure/Equation

In appendix and rebuttal,
if you want to set the beginning counting of figures/table/equations
to avoid conflict to the paper,
set these:

```tex
% Say you have 3 figures, 6 tables and 2 equations in the paper.
\setcounter{figure}{3} % new figure countings begin with 4, i.e. 3 + 1.
\setcounter{table}{6}
\setcounter{equation}{2}
```

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

# Quotation

Use `quote` or `quotation` environment.
One can add background colour to the quotation block by using the `tcolorbox` package:
```tex
\usepackage{tcolorbox}

\newenvironment{bgquote}
{
    \begin{tcolorbox}[
        colback=gray!10,          % Light gray background
        colframe=gray!10,         % Same color frame
        boxrule=0pt,             % No visible border
        arc=0pt,                 % Square corners
        left=0pt,               % Extra left padding
        right=0pt,              % Extra right padding
        top=5pt,                % Top padding
        bottom=5pt              % Bottom padding
    ]
    \begin{quote}
}
{\end{quote}\end{tcolorbox}}

\begin{document}
\begin{bgquote}
Quotation content.
\end{bgquote}
\end{document}
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

# Layout

To make the appendix be one-column in a two-column article, use `\onecolumn` command:

- ref: [Single Column Appendix in twocolumn article](https://tex.stackexchange.com/questions/230256/single-column-appendix-in-twocolumn-article)

```tex
\documentclass[twocolumn]{article} % two-column template
\begin{document}
Body here.

\appendix
\onecolumn % make the appendix section be one-column
\section{Foo Bar}
Appendix here.
\end{document}
```

Or one can use the `strip` environment provided by the [cuted](https://ctan.org/pkg/cuted) package to embed a one-column block. An example from [Magic Mirror: ID-Preserved Video Generation in Video Diffusion Transformers](https://arxiv.org/abs/2501.03931):

```tex
% \usepackage{cuted}

\begin{strip}
\begin{minipage}[b]{\textwidth}
    \centering
    \animategraphics[width=\linewidth,autoplay,loop,every=2]{6}{figs/gifs/output1_frames_blended/}{0}{47}
\end{minipage}%
\captionof{figure}{caption}
\end{strip}
```

# Cross Reference in Markdown/HTML

In HTML and markdown (e.g. your project page),
we sometimes want to refer to some location within the same site for easy navigation,
e.g. a section (title), figure or a table.
In LaTeX,
this is achieved by `\label` and `\ref`:
```tex
\section{Bar}
\label{foo} % set a label
% ...blabla...
As mentioned in \ref{foo}, % refer to that label
```
In HTML, this is realised by the `a` tag with its `name` attribute [8]:
```html
<h1>Bar</h1>
<a name="foo"></a> <!-- set a label -->
<!-- ...blabla... -->
As mentioned in <a href="#foo">Bar</a>, <!-- refer to this label -->
```
In markdown,
as HTML tags can be embedded into markdown directly,
one can set and refer to labels in the same way as in HTML.
But one can also refer with a simplified syntax:
```markdown
As mentioned in [Bar](#foo),
```

# Rebuttal Tracking Template

After receving the review comments,
we need to analyse them to direct our rebuttal.
See [Rebuttal Tracking Template.xlsx](Rebuttal_Tracking_Template.xlsx) for an analysing template.
Source: [GAMES003: 图形视觉科研基本素养](https://pengsida.net/games003/) -> Ninth week [Slides](https://pengsida.net/games003/GAMES003_files/week_9.pdf) -> [Rebuttal Tracking Template](https://docs.google.com/spreadsheets/d/1-FqA8RfQY5XwycJLqjVLLM0QIVNwzelGJqpWSInCen0/edit?usp=sharing).

# References

1. [latex空心小写字母、数字、括号](https://blog.csdn.net/HackerTom/article/details/134221777)
2. [latex自定义缩写](https://blog.csdn.net/HackerTom/article/details/134233080)
3. [latex cite命令、款式](https://blog.csdn.net/HackerTom/article/details/134318666)
4. [guanyingc/cv_rebuttal_template](https://github.com/guanyingc/cv_rebuttal_template)
5. [guanyingc/latex_paper_writing_tips](https://github.com/guanyingc/latex_paper_writing_tips)
6. [MLNLP-World/Paper-Writing-Tips](https://github.com/MLNLP-World/Paper-Writing-Tips)
7. [MLNLP-World/Paper-Picture-Writing-Code](https://github.com/MLNLP-World/Paper-Picture-Writing-Code)
8. [try writing blogs with GitHub issue #1](https://github.com/iTomxy/blogs/issues/1)
