PYTHON = uv run python

.PHONY: install run debug clean lint 

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for d in ('src', 'tests') for p in pathlib.Path(d).rglob('__pycache__')]; [shutil.rmtree(p, ignore_errors=True) for p in ['.mypy_cache', '.pytest_cache']]"

lint:
	uv run flake8 --exclude=.venv,venv .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --exclude='\.venv|venv'


