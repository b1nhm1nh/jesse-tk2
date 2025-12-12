# jesse-tk2 backtest

A streamlined backtest command for running Jesse strategies without needing `routes.py` configuration.

## Basic Usage

```bash
jesse-tk2 backtest STRATEGY START_DATE FINISH_DATE [OPTIONS]
```

### Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `STRATEGY` | Strategy class name (must exist in `strategies/` folder) | `Bot01_MCDX_SmartMoney` |
| `START_DATE` | Backtest start date in `YYYY-MM-DD` format | `2024-01-01` |
| `FINISH_DATE` | Backtest end date in `YYYY-MM-DD` format | `2024-12-31` |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--symbol` | `ETH-USDT` | Trading pair symbol (use dash format: `BTC-USDT`) |
| `--tf` | `1h` | Primary timeframe for the strategy |
| `--extra_tf` | `""` | Additional timeframes for multi-timeframe analysis |
| `--hp` | `""` | Hyperparameters as JSON dict |
| `--show-metrics/--no-show-metrics` | `True` | Display detailed metrics table |
| `--show-json-metrics/--no-show-json-metrics` | `False` | Display metrics as JSON |

## Examples

### Simple Backtest

```bash
jesse-tk2 backtest MyStrategy 2024-01-01 2024-06-30 --symbol BTC-USDT
```

### With Custom Timeframe

```bash
jesse-tk2 backtest MyStrategy 2024-01-01 2024-06-30 --symbol BTC-USDT --tf 4h
```

### With Hyperparameters

```bash
jesse-tk2 backtest MyStrategy 2024-01-01 2024-06-30 --symbol BTC-USDT --hp '{"ema_period": 20, "stop_loss": 5}'
```

## Multi-Timeframe Support

Use `--extra_tf` to provide additional timeframes your strategy needs. This is useful when your strategy analyzes multiple timeframes (e.g., using higher timeframe for trend and lower for entry).

### Single Extra Timeframe

```bash
jesse-tk2 backtest MyStrategy 2024-01-01 2024-06-30 --symbol BTC-USDT --tf 1h --extra_tf 4h
```

### Multiple Extra Timeframes

Separate timeframes with commas:

```bash
jesse-tk2 backtest MyStrategy 2024-01-01 2024-06-30 --symbol BTC-USDT --tf 15m --extra_tf "1h, 4h, 1D"
```

### Accessing Extra Timeframes in Strategy

In your strategy, access extra timeframe candles using Jesse's `get_candles()` method:

```python
def should_long(self):
    # Get 4h candles for higher timeframe analysis
    candles_4h = self.get_candles(self.exchange, self.symbol, '4h')

    # Get 1D candles for daily trend
    candles_1d = self.get_candles(self.exchange, self.symbol, '1D')

    # Your multi-timeframe logic here
    ...
```

## Supported Timeframes

| Timeframe | Code |
|-----------|------|
| 1 minute | `1m` |
| 3 minutes | `3m` |
| 5 minutes | `5m` |
| 15 minutes | `15m` |
| 30 minutes | `30m` |
| 45 minutes | `45m` |
| 1 hour | `1h` |
| 2 hours | `2h` |
| 3 hours | `3h` |
| 4 hours | `4h` |
| 6 hours | `6h` |
| 8 hours | `8h` |
| 12 hours | `12h` |
| 1 day | `1D` |

## Default Configuration

The backtest uses these default settings:

```python
{
    'starting_balance': 10_000,
    'fee': 0.0006,          # 0.06% (Binance Futures taker fee)
    'type': 'futures',
    'futures_leverage': 10,
    'futures_leverage_mode': 'cross',
    'exchange': 'Binance Perpetual Futures',
    'warm_up_candles': 100
}
```

## Warm-up Period

The backtest automatically calculates the required warm-up period based on:
- The largest timeframe used (primary + extra timeframes)
- The `warm_up_candles` setting (default: 100)

For example, if using `1D` timeframe with 100 warm-up candles, the system will load ~100 extra days of data before your start date.

## Output

The command displays a metrics table including:
- Total trades, win rate, net profit
- Sharpe, Sortino, Calmar ratios
- Max drawdown, annual return
- Winning/losing streaks
- Average win/loss amounts

Use `--show-json-metrics` for machine-readable JSON output.
