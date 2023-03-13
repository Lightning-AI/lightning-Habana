# Lightning + Intel Habana

[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://lightning.ai/)
[![PyPI Status](https://badge.fury.io/py/lightning-habana.svg)](https://badge.fury.io/py/lightning-habana)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lightning-habana)](https://pypi.org/project/lightning-habana/)
[![PyPI Status](https://pepy.tech/badge/lightning-habana)](https://pepy.tech/project/lightning-habana)
[![Deploy Docs](https://github.com/Lightning-AI/lightning-Habana/actions/workflows/docs-deploy.yml/badge.svg)](https://lightning-ai.github.io/lightning-Habana/)

[![General checks](https://github.com/Lightning-AI/lightning-habana/actions/workflows/ci-checks.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-habana/actions/workflows/ci-checks.yml)
[![Build Status](https://dev.azure.com/Lightning-AI/compatibility/_apis/build/status/Lightning-AI.lightning-Habana?branchName=main)](https://dev.azure.com/Lightning-AI/compatibility/_build/latest?definitionId=45&branchName=main)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Lightning-AI/lightning-Habana/main.svg)](https://results.pre-commit.ci/latest/github/Lightning-AI/lightning-Habana/main)

## To be Done aka cross-check

You still need to enable some external integrations such as:

- [ ] lock the main breach in GH setting - no direct push without PR
- [ ] init Read-The-Docs (add this new project)
- [ ] add credentials for releasing package to PyPI

## Tests / Docs notes

- We are using [Napoleon style,](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) and we shall use static types...
- It is nice to se [doctest](https://docs.python.org/3/library/doctest.html) as they are also generated as examples in documentation
- For wider and edge cases testing use [pytest parametrization](https://docs.pytest.org/en/stable/parametrize.html) :\]
