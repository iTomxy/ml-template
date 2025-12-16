# ml-template

My personal code snippet library of helper functions and classes, some with example.

- [+itom/](+itom): Tool functions that might be used in MATLAB codes.
The leading `+` will add this folder into the searching path.

- [configurations/](configurations): Configuration files.

- [containers/](containers): Definition files of Docker and Singularity image.

- [git/](git): git-related command showcases and *.gitignore* template.

- [latex/](latex): LaTeX preamble template for academic papers and a rebuttal analysis Excel template.

- [losses/](losses): Implementation of some deep learning loss functions.

- [scripts/](scripts): Utility scripts.

- [utils/](utils): Utilities.

- [args.py](args.py): An example of using the `argparse` package in Python.

- [config.yaml](config.yaml): An example of writing arguments in YAML.
Can be used together with [utils/config.py](utils/config.py).

- [requirements.txt](requirements.txt): Common packages I met that a Docker image may not contain.

# badge ![badge](https://img.shields.io/badge/badge-purple)

Sometimes you may want to create a badge in your GitHub repository README.
Use [Shields.io](https://github.com/badges/shields) to make one.
See [Static Badge](https://shields.io/badges/static-badge) for basic usage.
In markdown, you make one by inserting a:
```md
![SOME_ALTERNATIVE_TEXT](https://img.shields.io/badge/:badgeContent)
```
where `:badgeContent` has to be replaced by some fields.
For example,
a badge of this repository can be ![iTomxy/ml-template](https://img.shields.io/badge/iTomxy-ml--template-blue?logo=github&link=https%3A%2F%2Fgithub.com%2FiTomxy%2Fml-template):
```md
![iTomxy/ml-template](https://img.shields.io/badge/iTomxy-ml--template-blue?logo=github)
```
where:
- `iTomxy-ml--template-blue` is a `-`-separated string,
containing 3 fields: 1) left part text, 2) right part text, and 3) right part background colour.
- `logo` specifies the logo shown in the left part. The logo name should be from [Simple Icons](https://simpleicons.org/).

If you also want to add a hyperlink on the badge instead of letting it be a pure icon,
you just wrap it with `[badge-code](link)` [![iTomxy/ml-template](https://img.shields.io/badge/iTomxy-ml--template-blue?logo=github)](https://github.com/iTomxy/ml-template):
```md
[![iTomxy/ml-template](https://img.shields.io/badge/iTomxy-ml--template-blue?logo=github)](https://github.com/iTomxy/ml-template)
```
Althoug the Shilds.io badge supports inserting link by itself
(see [Static Badge](https://shields.io/badges/static-badge)),
but it seems that this is not usable in GitHub
(see [How to specify the link of left and right on GitHub #5593](https://github.com/badges/shields/discussions/5593)).
