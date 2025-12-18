"""
PairScan - Scan multiple pairs with a single strategy using the new parquet-based backtest.
Supports concurrent execution for faster scanning.
"""
import importlib
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
from time import strftime, gmtime
from timeit import default_timer as timer

import arrow
import numpy as np

from jessetk2.Vars import datadir

jessetkdir = datadir


def _get_candles(exchange: str, symbol: str, start_date: str, finish_date: str) -> np.ndarray:
    """Load candles using jesse research API (parquet files)."""
    from jesse.research import get_candles
    start_timestamp = arrow.get(start_date, 'YYYY-MM-DD').int_timestamp * 1000
    finish_timestamp = arrow.get(finish_date, 'YYYY-MM-DD').int_timestamp * 1000
    _, candles = get_candles(exchange, symbol, '1m', start_timestamp, finish_timestamp)
    return candles


def _convert_numpy_to_float(d):
    """Convert numpy types to Python native types."""
    for k, v in d.items():
        if isinstance(v, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            d[k] = int(v)
        elif isinstance(v, (np.float_, np.float16, np.float32, np.float64)):
            d[k] = round(float(v), 2)
        elif isinstance(v, float):
            d[k] = round(v, 2)
        if v == np.inf or v == np.nan or v is None:
            d[k] = 0
    return d


def _empty_metrics():
    """Return empty metrics dict for failed backtests."""
    return {
        'total': 0, 'total_winning_trades': 0, 'total_losing_trades': 0,
        'starting_balance': 0, 'finishing_balance': 0, 'win_rate': 0,
        'ratio_avg_win_loss': 0, 'longs_count': 0, 'longs_percentage': 0,
        'shorts_percentage': 0, 'shorts_count': 0, 'fee': 0, 'net_profit': 0,
        'net_profit_percentage': 0, 'average_win': 0, 'average_loss': 0, 'expectancy': 0,
        'expectancy_percentage': 0, 'expected_net_profit_every_100_trades': 0,
        'average_holding_period': 0, 'average_winning_holding_period': 0,
        'average_losing_holding_period': 0, 'gross_profit': 0, 'gross_loss': 0,
        'max_drawdown': 0, 'annual_return': 0, 'sharpe_ratio': 0, 'calmar_ratio': 0,
        'sortino_ratio': 0, 'omega_ratio': 0, 'serenity_index': 0, 'smart_sharpe': 0,
        'smart_sortino': 0, 'total_open_trades': 0, 'open_pl': 0, 'winning_streak': 0,
        'losing_streak': 0, 'largest_losing_trade': 0, 'largest_winning_trade': 0,
        'current_streak': 0
    }


def _run_single_backtest(params: dict) -> dict:
    """
    Run backtest for a single pair. This function runs in a separate process.
    """
    import jesse.helpers as jh
    from jesse.research.backtest import backtest

    symbol = params['symbol']
    strategy = params['strategy']
    exchange = params['exchange']
    timeframe = params['timeframe']
    start_date = params['start_date']
    finish_date = params['finish_date']
    warmup_start = params['warmup_start']
    config = params['config']
    hp = params['hp']

    routes = [
        {'exchange': exchange, 'strategy': strategy,
         'symbol': symbol, 'timeframe': timeframe}
    ]
    extra_routes = []

    try:
        candles = {
            jh.key(exchange, symbol): {
                'exchange': exchange,
                'symbol': symbol,
                'candles': _get_candles(exchange, symbol, warmup_start, finish_date)
            }
        }

        result = backtest(
            config,
            routes,
            extra_routes,
            candles,
            hyperparameters=hp
        )
        metrics = result['metrics']
        _convert_numpy_to_float(metrics)

        # Calculate market change
        candle_data = candles[jh.key(exchange, symbol)]['candles']
        first_close = candle_data[0][2]
        last_close = candle_data[-1][2]
        market_change = round(((last_close - first_close) / first_close) * 100.0, 2)
        metrics['market_change'] = market_change
        metrics['pair'] = symbol
        metrics['timeframe'] = timeframe
        metrics['start_date'] = start_date
        metrics['finish_date'] = finish_date
        metrics['error'] = None

        return metrics

    except Exception as e:
        metrics = _empty_metrics()
        metrics['market_change'] = 0
        metrics['pair'] = symbol
        metrics['timeframe'] = timeframe
        metrics['start_date'] = start_date
        metrics['finish_date'] = finish_date
        metrics['error'] = str(e)
        return metrics


class PairScan:
    def __init__(self, strategy: str, pairs_file: str, start_date: str, finish_date: str,
                 timeframe: str = '1h', exchange: str = 'Binance Perpetual Futures',
                 hp: dict = None, cpu: int = 1):
        import signal
        signal.signal(signal.SIGINT, self.signal_handler)

        self.strategy = strategy
        self.pairs_file = pairs_file
        self.start_date = start_date
        self.finish_date = finish_date
        self.timeframe = timeframe
        self.exchange = exchange
        self.hp = hp
        self.cpu = cpu

        self.pairs = []
        self.results = []
        self.sorted_results = []

        self.config = {
            'starting_balance': 10_000,
            'fee': 0.0006,
            'type': 'futures',
            'futures_leverage': 10,
            'futures_leverage_mode': 'cross',
            'exchange': exchange,
            'warm_up_candles': 100,
        }

        self.ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize exchange name for filename (replace spaces with underscores)
        exchange_safe = exchange.replace(' ', '_')
        self.filename = f'PairScan-{exchange_safe}-{timeframe}--{start_date}--{finish_date}'
        self.report_file_name = f'{jessetkdir}/results/{self.filename}--{self.ts}.csv'
        self.log_file_name = f'{jessetkdir}/logs/{self.filename}--{self.ts}.log'
        self.pairfile_name = f'{jessetkdir}/pairfiles/{self.filename}--{self.ts}.py'

    def signal_handler(self, sig, frame):
        print('\nYou pressed Ctrl+C!')
        self.save_results()
        sys.exit(0)

    def import_pairs(self):
        """Import pairs from the pairs file."""
        module_name = self.pairs_file.replace('\\', '.').replace('.py', '')
        module_name = module_name.replace('/', '.')
        print(f'Loading pairs from: {module_name}')

        pairs_module = importlib.import_module(module_name)
        importlib.reload(pairs_module)
        self.pairs = pairs_module.pairs
        print(f'Imported {len(self.pairs)} pairs')

    def _calculate_warmup(self):
        """Calculate warmup parameters."""
        import jesse.helpers as jh
        max_tf = jh.timeframe_to_one_minutes(self.timeframe)
        warmup_days = math.ceil(max_tf * self.config['warm_up_candles'] / 1440)
        warmup_start = arrow.get(self.start_date).shift(days=-warmup_days).format("YYYY-MM-DD")
        new_warm_up = math.floor(warmup_days * 1440 / max_tf)
        return warmup_start, new_warm_up

    def run(self):
        """Run backtest for all pairs with concurrent execution."""
        self.import_pairs()
        num_pairs = len(self.pairs)
        start = timer()

        # Calculate warmup once
        warmup_start, new_warm_up = self._calculate_warmup()
        config = deepcopy(self.config)
        config['warm_up_candles'] = new_warm_up

        print(f'\nScanning {num_pairs} pairs with {self.strategy} on {self.timeframe}')
        print(f'Period: {self.start_date} -> {self.finish_date}')
        print(f'Concurrent workers: {self.cpu}\n')

        # Prepare all backtest parameters
        all_params = []
        for pair in self.pairs:
            params = {
                'symbol': pair,
                'strategy': self.strategy,
                'exchange': self.exchange,
                'timeframe': self.timeframe,
                'start_date': self.start_date,
                'finish_date': self.finish_date,
                'warmup_start': warmup_start,
                'config': config,
                'hp': self.hp,
            }
            all_params.append(params)

        # Run backtests concurrently
        completed = 0
        with ProcessPoolExecutor(max_workers=self.cpu) as executor:
            # Submit all tasks
            future_to_pair = {executor.submit(_run_single_backtest, p): p['symbol'] for p in all_params}

            # Process results as they complete
            for future in as_completed(future_to_pair):
                pair = future_to_pair[future]
                completed += 1

                try:
                    metrics = future.result()
                    self.results.append(metrics)

                    # Sort by sharpe ratio
                    self.sorted_results = sorted(
                        self.results,
                        key=lambda x: float(x.get('sharpe_ratio', 0) or 0),
                        reverse=True
                    )

                    # Calculate ETA
                    elapsed = timer() - start
                    eta = (elapsed / completed) * (num_pairs - completed)
                    eta_formatted = strftime("%H:%M:%S", gmtime(eta))

                    sharpe = metrics.get('sharpe_ratio', 0) or 0
                    profit = metrics.get('net_profit_percentage', 0) or 0
                    trades = metrics.get('total', 0) or 0
                    error = metrics.get('error')

                    if error:
                        print(f'[{completed}/{num_pairs}] {pair}: ERROR - {error[:50]}')
                    else:
                        print(f'[{completed}/{num_pairs}] {pair}: Sharpe={sharpe:.2f} Profit={profit:.1f}% Trades={trades} ETA={eta_formatted}')

                except Exception as e:
                    print(f'[{completed}/{num_pairs}] {pair}: FAILED - {e}')

                # Print top results periodically
                if completed % (self.cpu * 2) == 0 or completed == num_pairs:
                    self.print_top_results()

        self.save_results()
        elapsed_total = timer() - start
        elapsed_formatted = strftime("%H:%M:%S", gmtime(elapsed_total))
        print(f'\nCompleted in {elapsed_formatted}')
        print(f'Results saved to:')
        print(f'  CSV: {self.report_file_name}')
        print(f'  Pairs: {self.pairfile_name}')

    def print_top_results(self):
        """Print top 15 results."""
        print('\n' + '=' * 100)
        print(f'{"Pair":<15} {"Sharpe":>8} {"Profit%":>10} {"MaxDD%":>8} {"WinRate%":>10} '
              f'{"Trades":>8} {"Calmar":>8} {"Serenity":>10} {"MktChg%":>8}')
        print('-' * 100)

        for r in self.sorted_results[:15]:
            pair = r.get('pair', '')[:14]
            sharpe = r.get('sharpe_ratio', 0) or 0
            profit = r.get('net_profit_percentage', 0) or 0
            max_dd = r.get('max_drawdown', 0) or 0
            win_rate = (r.get('win_rate', 0) or 0) * 100  # Convert to percentage
            trades = r.get('total', 0) or 0
            calmar = r.get('calmar_ratio', 0) or 0
            serenity = r.get('serenity_index', 0) or 0
            mkt_chg = r.get('market_change', 0) or 0

            print(f'{pair:<15} {sharpe:>8.2f} {profit:>10.1f} {max_dd:>8.1f} {win_rate:>10.1f} '
                  f'{trades:>8} {calmar:>8.2f} {serenity:>10.2f} {mkt_chg:>8.1f}')

        print('=' * 100 + '\n')

    def save_results(self):
        """Save results to CSV and pairs file."""
        if not self.sorted_results:
            print('No results to save')
            return

        # Save CSV report
        headers = ['Pair', 'Timeframe', 'Start', 'End', 'Trades', 'Profit%', 'MaxDD%',
                   'AnnualReturn%', 'WinRate%', 'Sharpe', 'Calmar', 'Sortino', 'Serenity',
                   'WinStreak', 'LoseStreak', 'LargestWin', 'LargestLoss', 'MarketChange%']

        with open(self.report_file_name, 'w', encoding='utf-8') as f:
            f.write('\t'.join(headers) + '\n')
            for r in self.sorted_results:
                row = [
                    r.get('pair', ''),
                    r.get('timeframe', ''),
                    r.get('start_date', ''),
                    r.get('finish_date', ''),
                    str(r.get('total', 0) or 0),
                    str(r.get('net_profit_percentage', 0) or 0),
                    str(r.get('max_drawdown', 0) or 0),
                    str(r.get('annual_return', 0) or 0),
                    str(round((r.get('win_rate', 0) or 0) * 100, 1)),  # Convert to percentage
                    str(r.get('sharpe_ratio', 0) or 0),
                    str(r.get('calmar_ratio', 0) or 0),
                    str(r.get('sortino_ratio', 0) or 0),
                    str(r.get('serenity_index', 0) or 0),
                    str(r.get('winning_streak', 0) or 0),
                    str(r.get('losing_streak', 0) or 0),
                    str(r.get('largest_winning_trade', 0) or 0),
                    str(r.get('largest_losing_trade', 0) or 0),
                    str(r.get('market_change', 0) or 0),
                ]
                f.write('\t'.join(row) + '\n')

        # Save sorted pairs file
        with open(self.pairfile_name, 'w', encoding='utf-8') as f:
            f.write('# Pairs sorted by Sharpe ratio\n')
            f.write(f'# Strategy: {self.strategy}\n')
            f.write(f'# Timeframe: {self.timeframe}\n')
            f.write(f'# Period: {self.start_date} -> {self.finish_date}\n\n')
            f.write('pairs = [\n')
            for r in self.sorted_results:
                sharpe = r.get('sharpe_ratio', 0) or 0
                profit = r.get('net_profit_percentage', 0) or 0
                f.write(f"    '{r.get('pair', '')}',  # Sharpe: {sharpe:.2f}, Profit: {profit:.1f}%\n")
            f.write(']\n')


def run(strategy: str, pairs_file: str, start_date: str, finish_date: str,
        timeframe: str = '1h', exchange: str = 'Binance Perpetual Futures',
        hp: dict = None, cpu: int = 1):
    """Entry point for pairscan command."""
    scanner = PairScan(
        strategy=strategy,
        pairs_file=pairs_file,
        start_date=start_date,
        finish_date=finish_date,
        timeframe=timeframe,
        exchange=exchange,
        hp=hp,
        cpu=cpu
    )
    scanner.run()
