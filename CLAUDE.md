# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Jesse-tk2 is a command-line toolkit that extends the [Jesse](https://jesse.trade) algorithmic trading framework. It provides advanced tools for:
- Bulk downloading historical candle data from Binance (spot and futures)
- Backtesting with hyperparameter injection (DNA, SEQ, HP formats)
- Parallel strategy refinement and walk-forward optimization
- Random walk testing for strategy robustness validation

## Installation

```bash
pip install -e .
```

The package installs as `jesse-tk2` CLI command.

## Key Commands

```bash
# Bulk download candles from Binance
jesse-tk2 bulk futures ETH-USDT 2020-01-01 --workers 8
jesse-tk2 bulk spot BTC-USDT 2020-01-01

# Backtest with hyperparameters
jesse-tk2 backtest StrategyName 2021-01-01 2022-01-01 --hp '{"param": value}'

# Refine DNAs (parallel backtesting)
jesse-tk2 refine dnafile.py 2021-01-01 2022-01-01 --cpu 8

# Refine sequential parameters
jesse-tk2 refine-seq params.txt 2021-01-01 2022-01-01 --cpu 8

# Random walk testing
jesse-tk2 random 2021-01-01 2022-01-01 40 60 --cpu 8

# Walk-forward optimization
jesse-tk2 walkforward dnafile.py 2020-01-01 2023-01-01 3 6 --cpu 8

# Pick best DNAs from optimization logs
jesse-tk2 pick optimization_log.txt
jesse-tk2 optuna-pick 0.001 -59.0
```

## Architecture

### Entry Point
- `jessetk2/__init__.py` - CLI entry point using Click. Defines all commands and handles route/config injection from the Jesse project directory.

### Core Modules
- `jessetk2/RefineTh.py`, `RefineSeq.py`, `RefineTh2.py` - Parallel DNA/parameter refinement using subprocess threading
- `jessetk2/RandomWalkTh.py` - Threaded random walk backtesting for robustness testing
- `jessetk2/BulkJesse.py`, `Bulk.py` - Bulk candle data downloading from Binance Vision
- `jessetk2/utils.py` - Console output parsing (`get_metrics3`), DNA encoding/decoding, file utilities
- `jessetk2/picker.py`, `OptunaPick.py` - DNA selection from optimization logs

### Hyperparameter Encoding Formats
1. **DNA** - Jesse's native base32 encoded format for integer hyperparameters
2. **SEQ** - Fixed-width zero-padded format. Example: `730900202` decodes as `[73, 09, 00, 20]` with width 2
3. **HP** - JSON dict format: `{"param_name": value}`

### Dependencies
The toolkit depends on `jessetk` (original jesse-toolkit) for some shared utilities like `Vars`, console formatters, and helpers.

## Important Patterns

### Project Validation
Commands require running from a valid Jesse project directory (containing `strategies/`, `config.py`, `storage/`, `routes.py`).

### Parallel Execution
Refinement classes spawn multiple `jesse-tk2 backtest` subprocesses and parse their console output to collect metrics. Results are sorted by Sharpe ratio by default.

### Output Files
Results are stored in `jessetkdata/`:
- `results/` - CSV reports
- `logs/` - Log files
- `dnafiles/` - Refined DNA files
- `pairfiles/` - Pair configuration files
