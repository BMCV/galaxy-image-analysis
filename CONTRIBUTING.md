# Contributing

This document describes how to contribute to this repository. Pull
requests containing bug fixes, updates, and extensions to the existing
tools and tool suites in this repository will be considered for
inclusion.

## How to Contribute

* Make sure you have a [GitHub account](https://github.com/signup/free)
* Make sure you have git [installed](https://help.github.com/articles/set-up-git)
* Fork the repository on [GitHub](https://github.com/BMCV/galaxy-image-analysis/fork)
* Make the desired modifications - consider using a [feature branch](https://github.com/Kunena/Kunena-Forum/wiki/Create-a-new-branch-with-git-and-manage-branches).
* Try to stick to the [Conventions for Tools in the Image Community](https://github.com/elixir-europe/biohackathon-projects-2023/blob/main/16/conventions.md) and the [IUC standards](http://galaxy-iuc-standards.readthedocs.org/en/latest/) whenever you can
* Make sure you have added the necessary tests for your changes and they pass.
* Open a [pull request](https://help.github.com/articles/using-pull-requests) with these changes.

## Using the new image-based verifications for tool testing

The new image-based verifications for Galaxy tool tests https://github.com/galaxyproject/galaxy/pull/17556 and https://github.com/galaxyproject/galaxy/pull/17581 won't be available in Galaxy before 24.1 is released.

Meanwhile, they are already available in the CI of the **galaxy-image-analyis** repostiroy! 🎉 https://github.com/BMCV/galaxy-image-analysis/pull/117

To also use them locally, you need to install the development versions of two Galaxy packages:
```python
python -m pip install git+https://git@github.com/galaxyproject/galaxy.git@5c1d045ce7b1e45f85608346baed5455324ee967#subdirectory=packages/util
python -m pip install git+https://git@github.com/galaxyproject/galaxy.git@5c1d045ce7b1e45f85608346baed5455324ee967#subdirectory=packages/tool_util
```

The commit [`5c1d045ce7b1e45f85608346baed5455324ee967`](https://github.com/galaxyproject/galaxy/commit/5c1d045ce7b1e45f85608346baed5455324ee967) corresponds to the latest merged bug fixes.

In addition, instead of running `planemo test`, you should use:
```python
planemo test --galaxy_source https://github.com/kostrykin/galaxy --galaxy_branch galaxy-image-analysis
```
The [galaxy-image-analysis branch](https://github.com/kostrykin/galaxy/tree/galaxy-image-analysis) of the <https://github.com/kostrykin/galaxy> fork is the same as the [23.1 release of Galaxy](https://github.com/galaxyproject/galaxy/tree/release_23.1), plus the support for the image-based verification extensions.
