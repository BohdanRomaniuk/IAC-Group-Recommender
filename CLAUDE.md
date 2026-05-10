# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Rules

- Do not add Claude as a co-author in commit messages. No `Co-Authored-By:` trailer.

## Commands

All commands run from `src/`.

```bash
# Install
pip install -r requirements.txt

# Generate datasets (run before first training)
python -m generators.generator_1m   # → src/data/MovieLens-1m/
python -m generators.generator_32m  # → src/data/MovieLens-32m/

# Train
python main.py
python main.py --config default.yaml
```

To switch to the 32m dataset, set `data_folder_path: ./data/MovieLens-32m` in your YAML.
