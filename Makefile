PYTHON = uv run python

.PHONY: install run debug clean lint 

install:
	uv sync

run:
	$(PYTHON) -m src

debug:
	$(PYTHON) -m pdb -m src

clean:
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for d in ('src', 'tests') for p in pathlib.Path(d).rglob('__pycache__')]; [shutil.rmtree(p, ignore_errors=True) for p in ['.mypy_cache', '.pytest_cache']]"

lint:
	uv run flake8 --exclude=.venv,venv,llm_sdk .
	uv run mypy src --follow-imports=silent --ignore-missing-imports --disable-error-code=import-untyped --disable-error-code=attr-defined --warn-return-any --warn-unused-ignores --disallow-untyped-defs --check-untyped-defs
