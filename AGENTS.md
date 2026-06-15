# Repository Guidelines

## Project Structure & Module Organization

This repository accompanies *Build a Large Language Model (From Scratch)* and is organized by book chapter. Top-level folders `ch01` through `ch07` and `appendix-A` through `appendix-E` contain chapter notebooks, scripts, datasets, and local READMEs. Main implementation material usually lives in `01_main-chapter-code`; optional experiments live in numbered bonus folders such as `ch05/03_bonus_pretraining_on_gutenberg`. Setup guidance is in `setup/`, with dependency checks in `setup/02_installing-python-libraries`. Tests are chapter-local `tests.py` files, not a single central test suite.

## Build, Test, and Development Commands

- `pip install -r requirements.txt`: install the core Python, PyTorch, Jupyter, and data dependencies used across chapters.
- `python setup/02_installing-python-libraries/python_environment_check.py`: verify the local Python environment after installation.
- `jupyter lab`: open notebooks for chapter walkthroughs and exercises.
- `pytest ch04/01_main-chapter-code/tests.py`: run a chapter test file; substitute the relevant `tests.py` path.
- `pytest --nbval ch03/01_main-chapter-code/multihead-attention.ipynb`: validate a notebook against saved outputs.
- `flake8 . --max-line-length=140 --ignore=W504,E402,E731,C406,E741,E722,E226`: mirror the CI style check.

## Coding Style & Naming Conventions

Use Python 3.10-compatible code and follow the existing educational style: clear names, small helper functions, and straightforward notebook-to-script correspondence. Use 4-space indentation. Keep Python module names lowercase with underscores where needed, following files such as `previous_chapters.py` and `gpt_train.py`. Preserve chapter folder naming patterns, including numbered prefixes like `01_main-chapter-code`.

## Testing Guidelines

The CI runs selected `pytest` files in `setup`, `ch04`, `ch05`, and `ch06`, plus selected notebooks through `nbval`. Add tests beside the chapter code they exercise and name test functions `test_*`. Keep tests lightweight enough for laptop execution; use reduced model sizes, small context lengths, and short epochs as current tests do. For notebooks, rerun cells before committing so `nbval` can compare stable outputs.

## Commit & Pull Request Guidelines

The current history uses short commit subjects. Prefer concise, imperative messages that identify the affected area, for example `Add ch06 classification smoke test`. Pull requests should include a short summary, changed chapter paths, commands run, and any data/model downloads required. Include screenshots only for notebook output or documentation rendering changes where visual review helps.

## Agent-Specific Instructions

Keep changes scoped to the requested chapter or setup area. Do not rewrite generated notebook output unless the task requires it, and avoid large downloaded model artifacts in commits.
