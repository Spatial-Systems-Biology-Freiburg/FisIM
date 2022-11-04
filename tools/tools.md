## Locally test workflows
Run act to test docker image locally:

```bash
sudo act --remote-name github --json -r --artifact-server-path /tmp/artifacts > logs.json
```

Afterwards split the result along the different jobs with custom script

```bash
python tools/analyze_docker.py
```

## Test Suite
```bash
python -m pytest -v
```

Filter for test with name containing "solv"

```bash
python -m pytest -v -k solv
```

## Test Coverage
We use coverage to measure the number of statements that were executed.

```bash
python -m coverage run -m pytest -v
python -m coverage report
```

To also display the line numbers which are missing run
```bash
python -m coverage report -m
```

## Testing symbolic differentiation
`symbolic_diff.py` is just for testing purposes

