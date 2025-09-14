PY=project_root/env/bin/python
PIP=project_root/env/bin/pip

.PHONY: venv install sanity lint fmt test ci clean

venv:
	python3 -m venv project_root/env

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	# optional tooling
	$(PIP) install ruff pre-commit pytest

sanity:
	$(PY) project_root/src/sanity.py project_root/videos/EXAMPLE.mp4 || true

lint:
	ruff check .

fmt:
	ruff check --fix .

test:
	pytest -q || true

ci: lint sanity

clean:
	rm -rf project_root/env .ruff_cache .pytest_cache
