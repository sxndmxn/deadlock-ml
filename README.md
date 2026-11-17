# deadlock-ml

`deadlock-ml` is an experimental Deadlock build analysis project.
Python scripts prepare match data and models.
A Rust web server displays item statistics, build paths, and related results.

## Requirements

The Python project requires Python 3.11 or newer and uv.
The current dependency set includes NVIDIA CUDA 12 packages for GPU processing.
The Rust server requires Cargo and a C++ compiler for its bundled DuckDB database library.

## Prepare data

Run these commands from the repository root:

```bash
uv sync --locked
uv run python scripts/fetch_data.py --help
uv run python ml/precompute.py --help
```

The fetch script downloads public Deadlock data into `data/`.
Select an available shard range with `--shards START-END` before downloading match data.
A shard is one part of the match dataset.
The precompute script reads that data and writes per-hero results under `models/`.
The repository does not include the complete match dataset or trained results.

## Run the server

Run Cargo from `rust-server/` after you prepare the data and model files:

```bash
cargo run --locked
```

Open `http://127.0.0.1:3000`.
The server reads `data/` and `models/` from the repository root.
Its static asset path requires `rust-server/` as the current directory.

## Related project

[deadlock-build-sync](https://github.com/sxndmxn/deadlock-build-sync) generates builds and installs them into Steam.
This repository provides a separate analysis and model training workflow.
