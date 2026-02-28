# Contributing to ORBIT

Thanks for your interest! Here's how to get started.

## Setup

```bash
git clone https://github.com/Rahillasne/Orbit.git
cd Orbit
pip install -e ".[dev]"
pytest tests/ -v
```

## Before submitting a PR

- Run `ruff check orbit/` and `ruff format orbit/`
- Run `pytest tests/ -v` and ensure all tests pass
- Add tests for new features

## Areas where help is welcome

- Additional failure detectors for new robot types
- LeRobot dataset integration improvements
- Dashboard visualizations
- Documentation and examples
