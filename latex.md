# Commonly Used Packages

```tex
% math
\usepackage{amsmath,amssymb,amsfonts}
% \usepackage{bm} % defines \bm
\usepackage{mathtools} % hollow char, see [1]
\usepackage{stmaryrd} % hollow bracket, see [1]

% pseudo code
\usepackage{algorithm}
\usepackage{algorithmic}

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
\usepackage{subcaption}

% text, colour
\usepackage{xcolor} % \textcolor, \definecolor
\usepackage{soul} % \st: stridethrough text

% space
\usepackage{xspace}

% url & cite
\definecolor{royalblue}{rgb}{0, 0.445, 0.737} % 0, 113, 188
\usepackage[pagebackref,breaklinks,colorlinks,bookmarks=true,citecolor=royalblue]{hyperref}
```

# Custom Commands

```tex
% abbreviations
\makeatletter
\DeclareRobustCommand\onedot{\futurelet\@let@token\@onedot}
\def\@onedot{\ifx\@let@token.\else.\null\fi\xspace}
\def\eg{\emph{e.g}\onedot} \def\Eg{\emph{E.g}\onedot}
\def\ie{\emph{i.e}\onedot} \def\Ie{\emph{I.e}\onedot}
\def\viz{\emph{viz}\onedot}
\def\cf{\emph{cf}\onedot} \def\Cf{\emph{Cf}\onedot}
\def\etc{\emph{etc}\onedot} \def\vs{\emph{vs}\onedot}
\def\wrt{w.r.t\onedot} \def\dof{d.o.f\onedot}
\def\iid{i.i.d\onedot} \def\wolog{w.l.o.g\onedot}
\def\etal{\emph{et al}\onedot}
\makeatother
% argmax, argmin
\DeclareMathOperator*{\argmin}{\arg\min}
\DeclareMathOperator*{\argmax}{\arg\max}
% iverson bracket
\DeclarePairedDelimiter{\iverson}{\llbracket}{\rrbracket}

% \bs := \boldsymbol
\ifdefined\bm
  \PackageWarning{\jobname}{\string\bm\space is already defined.}
\else
  \newcommand{\bm}{\boldsymbol}
\fi
```

# Custom Colours

You can use these colours to configure the citation, link and url colour of `hyperref`.

```tex
\definecolor{royalblue}{rgb}{0, 0.445, 0.737} % 0, 113, 188
\definecolor{bilibili}{rgb}{0.983, 0.446, 0.6} % 251, 114, 153
```

# References

1. [latex空心小写字母、数字、括号](https://blog.csdn.net/HackerTom/article/details/134221777)
2. [latex自定义缩写](https://blog.csdn.net/HackerTom/article/details/134233080)
3. [latex cite命令、款式](https://blog.csdn.net/HackerTom/article/details/134318666)
4. [guanyingc/cv_rebuttal_template](https://github.com/guanyingc/cv_rebuttal_template)
5. [guanyingc/latex_paper_writing_tips](https://github.com/guanyingc/latex_paper_writing_tips)
6. [MLNLP-World/Paper-Writing-Tips](https://github.com/MLNLP-World/Paper-Writing-Tips)
7. [MLNLP-World/Paper-Picture-Writing-Code](https://github.com/MLNLP-World/Paper-Picture-Writing-Code)
