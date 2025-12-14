# from email.policy import default
from uuid import RESERVED_FUTURE
import numpy as np
import os
import sys
import pprint as pp
import datetime as dt

import click
import jesse.helpers as jh
import math as math

# Python version validation.
if jh.python_version() < (3,7):
    print(
        jh.color(
            f'Jesse requires Python version above 3.7. Yours is {jh.python_version()}',
            'red'
        )
    )

# fix directory issue
sys.path.insert(0, os.getcwd())

ls = os.listdir('.')
is_jesse_project = 'strategies' in ls and 'config.py' in ls and 'storage' in ls and 'routes.py' in ls


def inject_local_config() -> None:
    """
    injects config from local config file
    """
    local_config = locate('config.config')
    from jesse.config import set_config
    set_config(local_config)


def inject_local_routes() -> None:
    """
    injects routes from local routes folder
    """
    local_router = locate('routes')
    from jesse.routes import router

    router.set_routes(local_router.routes)
    router.set_extra_candles(local_router.extra_candles)

# inject local files
if is_jesse_project:
    inject_local_config()
    inject_local_routes()


def validate_cwd() -> None:
    """
    make sure we're in a Jesse project
    """
    if not is_jesse_project:
        print(
            jh.color(
                'Current directory is not a Jesse project. You must run commands from the root of a Jesse project.',
                'red'
            )
        )
        # os.exit(1)
        exit()


# create a Click group

@click.group()
def cli() -> None:
    pass



# --------------------------------------------------
from contextlib import contextmanager
from timeit import default_timer

@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start


# --------------------------------------------------
from contextlib import contextmanager
from timeit import default_timer

@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start, 2

def _get_candles_with_cache(exchange: str, symbol: str, start_date: str, finish_date: str) -> np.ndarray:
    from jesse.research import get_candles
    import arrow
    # Convert string dates to timestamps
    start_timestamp = arrow.get(start_date, 'YYYY-MM-DD').int_timestamp * 1000
    finish_timestamp = arrow.get(finish_date, 'YYYY-MM-DD').int_timestamp * 1000
    # get_candles returns (warmup_candles, trading_candles) tuple
    _, candles = get_candles(exchange, symbol, '1m', start_timestamp, finish_timestamp)
    return candles

    path = pathlib.Path('storage/bulk')
    path.mkdir(parents=True, exist_ok=True)

    cache_file_name = f"{exchange}-{symbol}-1m-{start_date}-{finish_date}.pickle"
    cache_file = pathlib.Path(f'storage/bulk/{cache_file_name}')

    if cache_file.is_file():
        with open(f'storage/bulk/{cache_file_name}', 'rb') as handle:
            candles = pickle.load(handle)
    else:
        candles = get_candles(exchange, symbol, '1m', start_date, finish_date)
        with open(f'storage/bulk/{cache_file_name}', 'wb') as handle:
            pickle.dump(candles, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return candles

def _run_backtest(params):
    StrategyClass = jh.get_strategy_class(params['routes'][0]['strategy'])
    # hp_dict = StrategyClass().hyperparameters()
    # from importlib import reload

    # import jesse

    # import strategies
    # from pydoc import locate
    
    # reload(locate('strategies.' + params['routes'][0]['strategy']))
    import jesse.helpers as jh
    from jesse.strategies import Strategy
    from jesse import utils
    from jesse.research import candles_from_close_prices, get_candles
    from jesse.research.backtest import backtest

    exchange_name = params['exchange_name']
    symbol = params['symbol']
    config = params['config']
    routes = params['routes']
    extra_routes = params['extra_routes']
    hp = []

    candles = {
        # keys must be in this format: 'Fake Exchange-BTC-USDT'
        jh.key(exchange_name, symbol): {
            'exchange': exchange_name,
            'symbol': symbol,
            'candles': get_candles_with_cache(exchange_name, symbol, params['start_date'], params['finish_date'])
        },
    }

    got_exception = False
    try:
        backtest_data = backtest(config, routes, extra_routes, candles, hyperparameters = hp)['metrics']
        print("Done~~~")
        print(backtest_data)
        print(backtest_data['metrics'])
    except Exception as e:
        print("Exception: ", e)
        got_exception = True
            # logger = start_logger_if_necessary()
            # logger.error("".join(traceback.TracebackException.from_exception(e).format()), extra={'key': key})
            # Re-raise the original exception so the Pool worker can
            # clean up
    if got_exception or backtest_data['total'] == 0:
        backtest_data = {'total': 0, 'total_winning_trades': None, 'total_losing_trades': None,
                        'starting_balance': None, 'finishing_balance': None, 'win_rate': None,
                        'ratio_avg_win_loss': None, 'longs_count': None, 'longs_percentage': None,
                        'shorts_percentage': None, 'shorts_count': None, 'fee': None, 'net_profit': None,
                        'net_profit_percentage': None, 'average_win': None, 'average_loss': None, 'expectancy': None,
                        'expectancy_percentage': None, 'expected_net_profit_every_100_trades': None,
                        'average_holding_period': None, 'average_winning_holding_period': None,
                        'average_losing_holding_period': None, 'gross_profit': None, 'gross_loss': None,
                        'max_drawdown': None, 'annual_return': None, 'sharpe_ratio': None, 'calmar_ratio': None,
                        'sortino_ratio': None, 'omega_ratio': None, 'serenity_index': None, 'smart_sharpe': None,
                        'smart_sortino': None, 'total_open_trades': None, 'open_pl': None, 'winning_streak': None,
                        'losing_streak': None, 'largest_losing_trade': None, 'largest_winning_trade': None,
                        'current_streak': None}
    return backtest_data
    {
            'metrics': {'total': 0}
    }

def _convert_numpy_to_float(d):
    for k, v in d.items():
        if isinstance(v, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            d[k] = int(v)
        elif isinstance(v, (np.float_, np.float16, np.float32, np.float64)):
            d[k] = round(float(v),2)
        elif isinstance(v, float):
            d[k] = round(v,2)
        if v == np.inf or v == np.nan or v == None:
            d[k] = 0
    return d

@cli.command()
@click.argument('strategy', required=True, type=str)
@click.argument('start_date', required=True, type=str)
@click.argument('finish_date', required=True, type=str)
@click.option(
    '--symbol', default='ETH-USDT', show_default=True, help='Trading pair symbol (e.g., BTC-USDT)')
@click.option(
    '--hp', default='', show_default=True, help='Hyperparameters payload as dict')
@click.option(
    '--tf', default='1h', show_default=True, help='Timeframe')
@click.option(
    '--extra_tf', default='', show_default=True, help='Extra Timeframes "[1h, 4h, 1D]"')
@click.option(
    '--show-metrics/--no-show-metrics', default=True,
    help='Displays detailed Metric.'
)
@click.option(
    '--show-json-metrics/--no-show-json-metrics', default=False,
    help='Displays Json Metric.'
)
def backtest(strategy: str, start_date: str, finish_date: str, symbol: str, hp: str, tf: str, extra_tf: str, show_metrics: bool, show_json_metrics: bool)->None:
    import arrow
    import json

    import jessetk2.table as table
    from jessetk2.report import portfolio_metrics

    # print("Importing lib")
    with elapsed_timer() as elapsed:
        import jesse.helpers as jh
        from jesse.strategies import Strategy
        from jesse import utils
        from jesse.research import backtest, candles_from_close_prices
        # print("Elapsed: ", round(elapsed(),2), ' seconds')


    # print ("Loading candles...")

    exchange_name = 'Binance Perpetual Futures'
    timeframe = tf
    # '1h'
    config = {
        'starting_balance': 10_000,
        'fee': 0.0006,
        # accepted values are 'spot' and 'futures'
        'type': 'futures',
        # only used if type is 'futures'
        'futures_leverage': 10,
        # only used if type is 'futures'
        'futures_leverage_mode': 'cross',
        'exchange': exchange_name,
        'warm_up_candles': 100,
    }
    routes = [
        {'exchange': exchange_name, 'strategy': strategy, 'symbol': symbol, 'timeframe': timeframe}
    ]

    extra_routes = []
    if len(extra_tf):
        # extra_tf = extra_tf[1:-1]  # trim '[' and ']'
        extra_tfs = extra_tf.split(',') 
        for etf in extra_tfs:
            extra_routes.append({'exchange': exchange_name, 'strategy': strategy, 'symbol': symbol, 'timeframe': etf.strip()})

    max_tf = 1
    for route in routes:
        max_tf = max(max_tf, jh.timeframe_to_one_minutes(route['timeframe']))
    for route in extra_routes:
        max_tf = max(max_tf, jh.timeframe_to_one_minutes(route['timeframe']))
    # print(f"Max timeframe: ", max_tf)
    warmup_requirement_days = math.ceil(max_tf * config['warm_up_candles'] / 1440)
    # print(f"Warm-up days: ", warmup_requirement_days)
    # start_date_arrow = jh.date_to_timestamp(start_date)
    backtest_date = start_date
    start_date = arrow.get(start_date).shift(days=-warmup_requirement_days).format("YYYY-MM-DD")
    # start_date = jh.timestamp_to_date(jh.date_to_timestamp(start_date) - warmup_requirement_days * 14400000)
    new_warm_up = math.floor(warmup_requirement_days * 1440 / max_tf)
    config['warm_up_candles'] = new_warm_up

    # print("Real Start Date", start_date, 'warm up ',new_warm_up)

    candles = {
        jh.key(exchange_name, symbol): {
            'exchange': exchange_name,
            'symbol': symbol,
            'candles': _get_candles_with_cache(exchange_name, symbol, start_date, finish_date)
        },
    }
    # jh.timeframe_to_one_minutes
    hps = None
    if len(hp) > 0:
        hps = json.loads(hp)

    print (f"Start backtesting. Warm up: {start_date} | {backtest_date} -> {finish_date}. Timeframe: {timeframe}. Max timeframe: {max_tf}...")
    with elapsed_timer() as elapsed:
        got_exception = False
        try:
            result = backtest(
                config,
                routes,
                extra_routes,
                candles,
                hyperparameters=hps
            )
            # print(f"Result {result}")
            metrics = result['metrics']
            _convert_numpy_to_float(metrics)
        except Exception as e:
            import traceback
            traceback.print_exc()
            # print("Backtest Error Exception: ", e)
            got_exception = True

        if got_exception or metrics['total'] == 0:
            # result: dict = []
            metrics = {'total': 0, 'total_winning_trades': 0, 'total_losing_trades': 0,
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
                            'current_streak': 0}                         
        if show_json_metrics:
            print("JSON Metrics|", json.dumps(metrics))
 
        if show_metrics:
            first_close = candles[jh.key(exchange_name, symbol)]['candles'][0][2]
            last_close = candles[jh.key(exchange_name, symbol)]['candles'][-1][2]
            change = ((last_close - first_close) / first_close) * 100.0
            diff = candles[jh.key(exchange_name, symbol)]['candles'][-1][0] - candles[jh.key(exchange_name, symbol)]['candles'][-2][0]
            metrics['timeframe'] = jh.timeframe_to_one_minutes(timeframe) * 60
            data = portfolio_metrics(metrics)
            data.append(['Market Change', f"{str(round(change, 2))}%"])        
            table.key_value(data, 'Metrics', alignments=('left', 'right'))

        print("Elapsed: ", round(elapsed(),2), ' seconds')


def print_initial_msg():
    print(initial_test_message)


def makedirs():
    from jessetk2.Vars import datadir

    os.makedirs(f'./{datadir}', exist_ok=True)
    os.makedirs(f'./{datadir}/results', exist_ok=True)
    os.makedirs(f'./{datadir}/logs', exist_ok=True)
    os.makedirs(f'./{datadir}/dnafiles', exist_ok=True)
    os.makedirs(f'./{datadir}/pairfiles', exist_ok=True)


def validateconfig():
    pass
    # if not (get_config('env.metrics.sharpe_ratio', False) and
    #         get_config('env.metrics.calmar_ratio', False) and
    #         get_config('env.metrics.winning_streak', False) and
    #         get_config('env.metrics.losing_streak', False) and
    #         get_config('env.metrics.largest_losing_trade', False) and
    #         get_config('env.metrics.largest_winning_trade', False) and
    #         get_config('env.metrics.total_winning_trades', False) and
    #         get_config('env.metrics.total_losing_trades', False)):
    #     print('Set optional metrics to True in config.py!')
    #     exit()

# @cli.command()
# @click.argument('dna_file', required=True, type=str)
# @click.argument('start_date', required=True, type=str)
# @click.argument('finish_date', required=True, type=str)
# @click.argument('eliminate', required=False, type=bool)
# def refine(dna_file, start_date: str, finish_date: str, eliminate: bool) -> None:
#     """
#     backtest all candidate dnas. Enter in "YYYY-MM-DD" "YYYY-MM-DD"
#     """
#     os.chdir(os.getcwd())
#     validate_cwd()
#     validateconfig()
#     makedirs()

#     if not eliminate:
#         eliminate = False

#     from jessetk2.refine import refine
#     r = refine(dna_file, start_date, finish_date, eliminate)
#     r.run(dna_file, start_date, finish_date)


@cli.command()
@click.argument('dna_file', required=True, type=str)
@click.argument('start_date', required=True, type=str)
@click.argument('finish_date', required=True, type=str)
@click.argument('inc_month', required=True, type=int)
@click.argument('test_month', required=True, type=int)
@click.option('--eliminate/--no-eliminate', default=False,
              help='Remove worst performing dnas at every iteration.')
@click.option(
    '--dnas', default=160, show_default=True,
    help='Number of max dnas to test.')
@click.option(
    '--passno', default=1, show_default=True,
    help='Start-up pass number.')
@click.option(
    '--cpu', default=0, show_default=True,
    help='The number of CPU cores that Jesse is allowed to use. If set to 0, it will use as many as is available on your machine.')
@click.option(
    '--debug/--no-debug', default=False,
    help='Displays detailed logs about the genetics algorithm. Use it if you are interested int he genetics algorithm.'
)
@click.option('--csv/--no-csv', default=False, help='Outputs a CSV file of all DNAs on completion.')
@click.option('--json/--no-json', default=False, help='Outputs a JSON file of all DNAs on completion.')
def walkforward(dna_file: str, start_date: str, finish_date: str, inc_month : int, test_month: int, cpu: int, dnas: int, passno: int, eliminate:bool, debug: bool, csv: bool,
             json: bool) -> None:
    """
    Walkforward in period. Enter in "dna_file" "YYYY-MM-DD" "YYYY-MM-DD" "3" "6"
    """
    import arrow
    os.chdir(os.getcwd())
    validate_cwd()
    validateconfig()
    makedirs()

    if not eliminate:
        eliminate = False

    if cpu > cpu_count():
        raise ValueError(
            f'Entered cpu cores number is more than available on this machine which is {cpu_count()}')
    elif cpu == 0:
        max_cpu = cpu_count()
    else:
        max_cpu = cpu
    print('CPU:', max_cpu)

    from jessetk2.walk_forward import Refine


    print (f" Walkforward period: {start_date.format('YYYY-MM-DD')} - {finish_date.format('YYYY-MM-DD')}")

    a_start_date = arrow.get(start_date, 'YYYY-MM-DD')
    a_finish_date = arrow.get(finish_date, 'YYYY-MM-DD')
    i_start_date = a_start_date
    i_finish_date = i_start_date.shift(months = test_month)
    # passno = 1
    while  i_start_date <= a_finish_date:
        if i_finish_date > a_finish_date:
            i_finish_date = a_finish_date
        print (f"Walk {i_start_date.format('YYYY-MM-DD')} - {i_finish_date.format('YYYY-MM-DD')} ")
        r = Refine(dna_file, i_start_date.format('YYYY-MM-DD'), i_finish_date.format('YYYY-MM-DD'), dnas, eliminate, max_cpu, passno)
        dna_file = r.run()

        # calculate next period
        i_start_date = i_start_date.shift(months = inc_month)
        i_finish_date = i_start_date.shift(months = test_month)
        passno += 1


@cli.command()
@click.argument('treshold1', required=False, type=float, default=0.001)
@click.argument('treshold2', required=False, type=float, default=-59.0)
def optuna_pick(treshold1: float, treshold2:float) -> None:
    
    os.chdir(os.getcwd())
    validate_cwd()

    print(f"treshold1: {treshold1}")
    print(f"treshold2: {treshold2}")

    from jessetk2.OptunaPick import OptunaPick
    op = OptunaPick(t1=treshold1, t2=treshold2)
    op.dump_best_parameters()


@cli.command()
@click.argument('dna_log_file', required=True, type=str)
@click.argument('sort_criteria', required=False, type=str)
@click.argument('len1', required=False, type=int)
@click.argument('len2', required=False, type=int)
def pick(dna_log_file, sort_criteria, len1, len2) -> None:
    """
    Picks dnas from Jesse optimization log file
    """

    if not dna_log_file:
        print('dna_log_file is required!')
        exit()

    sort_criteria = 'pnl1' if not sort_criteria else sort_criteria
    len1 = 30 if not len1 or len1 < 0 or len1 > 10_000 else len1
    len2 = 150 if not len2 or len2 < 0 or len2 > 10_000 else len2

    os.chdir(os.getcwd())
    validate_cwd()

    import jesse.helpers as jh
    from jesse.routes import router

    makedirs()
    r = router.routes[0]  # Read first route from routes.py
    strategy = r.strategy_name
    StrategyClass = jh.get_strategy_class(r.strategy_name)
    print('Strategy name:', strategy, 'Strategy Class:', StrategyClass)

    from jessetk2.picker import picker

    dna_picker = picker(dna_log_file, strategy,
                        StrategyClass, len1, len2, sort_criteria)

    dna_picker.sortdnas()
    dna_picker.create_output_file()
    dna_picker.validate_output_file()


@cli.command()
@click.argument('dna_log_file', required=True, type=str)
@click.argument('sort_criteria', required=False, type=str)
@click.argument('len1', required=False, type=int)
@click.argument('len2', required=False, type=int)
def pick_csv(dna_log_file, sort_criteria, len1, len2) -> None:
    """
    Picks dnas from Jesse optimization csv log file
    """

    if not dna_log_file:
        print('dna_log_file is required!')
        exit()

    sort_criteria = 'pnl1' if not sort_criteria else sort_criteria
    len1 = 30 if not len1 or len1 < 0 or len1 > 10_000 else len1
    len2 = 150 if not len2 or len2 < 0 or len2 > 10_000 else len2

    os.chdir(os.getcwd())
    validate_cwd()

    import jesse.helpers as jh
    from jesse.routes import router

    makedirs()
    r = router.routes[0]  # Read first route from routes.py
    strategy = r.strategy_name
    StrategyClass = jh.get_strategy_class(r.strategy_name)
    print('Strategy name:', strategy, 'Strategy Class:', StrategyClass)

    from jessetk2.picker_csv import picker

    dna_picker = picker(dna_log_file, strategy,
                        StrategyClass, len1, len2, sort_criteria)

    dna_picker.sortdnas()
    # dna_picker.create_output_file()
    # dna_picker.validate_output_file()

@cli.command()
@click.argument('dna_file', required=True, type=str)
@click.argument('start_date', required=True, type=str)
@click.argument('finish_date', required=True, type=str)
@click.option('--eliminate/--no-eliminate', default=False,
              help='Remove worst performing dnas at every iteration.')
@click.option(
    '--cpu', default=0, show_default=True,
    help='The number of CPU cores that Jesse is allowed to use. If set to 0, it will use as many as is available on your machine.')
@click.option('--full-reports/--no-full-reports', default=False,
              help="Generates QuantStats' HTML output with metrics reports like Sharpe ratio, Win rate, Volatility, etc., and batch plotting for visualizing performance, drawdowns, rolling statistics, monthly returns, etc.")
def refine(dna_file, start_date: str, finish_date: str, eliminate: bool, cpu: int, full_reports) -> None:
    """
    backtest all candidate dnas. Enter in "YYYY-MM-DD" "YYYY-MM-DD"
    """
    os.chdir(os.getcwd())
    validate_cwd()
    validateconfig()
    makedirs()

    if not eliminate:
        eliminate = False

    if cpu > cpu_count():
        raise ValueError(
            f'Entered cpu cores number is more than available on this machine which is {cpu_count()}')
    elif cpu == 0:
        max_cpu = cpu_count()
    else:
        max_cpu = cpu
    print('CPU:', max_cpu)

    from jessetk2.RefineTh import Refine
    r = Refine(dna_file, start_date, finish_date,
               eliminate, max_cpu, full_reports)
    r.run()

@cli.command()
@click.argument('dna_file', required=True, type=str)
@click.argument('start_date', required=True, type=str)
@click.argument('finish_date', required=True, type=str)
@click.option(
    '--dnas', default=160, show_default=True,
    help='The number of Max DNA')
@click.option('--eliminate/--no-eliminate', default=False,
              help='Remove worst performing dnas at every iteration.')
@click.option(
    '--cpu', default=0, show_default=True,
    help='The number of CPU cores that Jesse is allowed to use. If set to 0, it will use as many as is available on your machine.')
def refine2(dna_file, start_date: str, finish_date: str, dnas:int, eliminate: bool, cpu: int) -> None:
    """
    backtest all candidate dnas. Enter in "YYYY-MM-DD" "YYYY-MM-DD"
    """
    os.chdir(os.getcwd())
    validate_cwd()
    validateconfig()
    makedirs()

    if not eliminate:
        eliminate = False

    if cpu > cpu_count():
        raise ValueError(
            f'Entered cpu cores number is more than available on this machine which is {cpu_count()}')
    elif cpu == 0:
        max_cpu = cpu_count()
    else:
        max_cpu = cpu
    print('CPU:', max_cpu)

    from jessetk2.RefineTh2 import Refine
    r = Refine(dna_file, start_date, finish_date, dnas, eliminate, max_cpu)
    r.run()

@cli.command()
@click.argument('dna_file', required=True, type=str)
@click.argument('start_date', required=True, type=str)
@click.argument('finish_date', required=True, type=str)
@click.option(
    '--dnas', default=160, show_default=True,
    help='The number of Max DNA')
@click.option('--eliminate/--no-eliminate', default=False,
              help='Remove worst performing dnas at every iteration.')
@click.option(
    '--cpu', default=0, show_default=True,
    help='The number of CPU cores that Jesse is allowed to use. If set to 0, it will use as many as is available on your machine.')
def refine_top(dna_file, start_date: str, finish_date: str, dnas: int, eliminate: bool, cpu: int) -> None:
    """
    backtest all candidate dnas. Enter in "YYYY-MM-DD" "YYYY-MM-DD"
    """
    os.chdir(os.getcwd())
    validate_cwd()
    validateconfig()
    makedirs()

    if not eliminate:
        eliminate = False

    if cpu > cpu_count():
        raise ValueError(
            f'Entered cpu cores number is more than available on this machine which is {cpu_count()}')
    elif cpu == 0:
        max_cpu = cpu_count()
    else:
        max_cpu = cpu
    print('CPU:', max_cpu)

    from jessetk2.refine_top import Refine
    r = Refine(dna_file, start_date, finish_date, dnas, eliminate, max_cpu)
    r.run()

@cli.command()
@click.argument('long_dna_file', required=True, type=str)
@click.argument('short_dna_file', required=True, type=str)
@click.argument('start_date', required=True, type=str)
@click.argument('finish_date', required=True, type=str)
@click.option(
    '--dnas', default=160, show_default=True,
    help='The number of Max DNA')
@click.option('--eliminate/--no-eliminate', default=False,
              help='Remove worst performing dnas at every iteration.')
@click.option(
    '--cpu', default=0, show_default=True,
    help='The number of CPU cores that Jesse is allowed to use. If set to 0, it will use as many as is available on your machine.')
def refinels(long_dna_file:str, short_dna_file: str, start_date: str, finish_date: str, dnas: int, eliminate: bool, cpu: int) -> None:
    """
    backtest all candidate dnas. Enter in "YYYY-MM-DD" "YYYY-MM-DD"
    """
    os.chdir(os.getcwd())
    validate_cwd()
    validateconfig()
    makedirs()

    if not eliminate:
        eliminate = False

    if cpu > cpu_count():
        raise ValueError(
            f'Entered cpu cores number is more than available on this machine which is {cpu_count()}')
    elif cpu == 0:
        max_cpu = cpu_count()
    else:
        max_cpu = cpu
    print('CPU:', max_cpu)

    from jessetk2.refine_long_short import Refine
    r = Refine(long_dna_file, short_dna_file, start_date, finish_date, dnas, eliminate, max_cpu)
    r.run()


@cli.command()
@click.argument('hp_file', required=True, type=str)
@click.argument('start_date', required=True, type=str)
@click.argument('finish_date', required=True, type=str)
@click.option(
    '--cpu', default=0, show_default=True,
    help='The number of CPU cores that Jesse is allowed to use. If set to 0, it will use as many as is available on your machine.')
def refine_hp(hp_file:str, start_date: str, finish_date: str, cpu: int) -> None:
    """
    backtest all candidate hp. Enter in "YYYY-MM-DD" "YYYY-MM-DD"
    """
    os.chdir(os.getcwd())
    validate_cwd()
    validateconfig()
    makedirs()

    if cpu > cpu_count():
        raise ValueError(
            f'Entered cpu cores number is more than available on this machine which is {cpu_count()}')
    elif cpu == 0:
        max_cpu = cpu_count()
    else:
        max_cpu = cpu
    print('CPU:', max_cpu)

    from jessetk2.refine_hp import Refine
    r = Refine(hp_file, start_date, finish_date, max_cpu)
    r.run()
@cli.command()
@click.argument('hp_file', required=True, type=str)
@click.argument('start_date', required=True, type=str)
@click.argument('finish_date', required=True, type=str)
@click.option('--eliminate/--no-eliminate', default=False,
              help='Remove worst performing dnas at every iteration.')
@click.option(
    '--cpu', default=0, show_default=True,
    help='The number of CPU cores that Jesse is allowed to use. If set to 0, it will use as many as is available on your machine.')
@click.option(
    '--dd', default=-90, show_default=True,
    help='Maximum drawdown limit for filtering results. Use negative values.')
@click.option('--full-reports/--no-full-reports', default=False,
              help="Generates QuantStats' HTML output with metrics reports like Sharpe ratio, Win rate, Volatility, etc., and batch plotting for visualizing performance, drawdowns, rolling statistics, monthly returns, etc.")
def refine_seq(hp_file, start_date: str, finish_date: str, eliminate: bool, cpu: int, dd: int, full_reports) -> None:
    """
    backtest all Sequential candidate Optuna parameters.
    Enter in "YYYY-MM-DD" "YYYY-MM-DD"
    """
    os.chdir(os.getcwd())
    validate_cwd()
    validateconfig()
    makedirs()

    if not eliminate:
        eliminate = False

    if cpu > cpu_count():
        raise ValueError(
            f'Entered cpu cores number is more than available on this machine which is {cpu_count()}')
    elif cpu == 0:
        max_cpu = cpu_count()
    else:
        max_cpu = cpu
    print('CPU:', max_cpu)

    from jessetk2.RefineSeq import Refine
    r = Refine(hp_file, start_date, finish_date, eliminate,
               max_cpu, dd=dd, full_reports=full_reports)
    r.run()

# @cli.command()
# @click.argument('dna_file', required=True, type=str)
# @click.argument('start_date', required=True, type=str)
# @click.argument('finish_date', required=True, type=str)
# @click.option('--eliminate/--no-eliminate', default=False,
#               help='Remove worst performing dnas at every iteration.')
# @click.option(
#     '--cpu', default=0, show_default=True,
#     help='The number of CPU cores that Jesse is allowed to use. If set to 0, it will use as many as is available on your machine.')
# def refine_gly(dna_file, start_date: str, finish_date: str, eliminate: bool, cpu: int) -> None:
#     """
#     backtest all candidate dnas. Enter in "YYYY-MM-DD" "YYYY-MM-DD"
#     """
#     os.chdir(os.getcwd())
#     validate_cwd()
#     validateconfig()
#     makedirs()

#     if not eliminate:
#         eliminate = False

#     if cpu > cpu_count():
#         raise ValueError(
#             f'Entered cpu cores number is more than available on this machine which is {cpu_count()}')
#     elif cpu == 0:
#         max_cpu = cpu_count()
#     else:
#         max_cpu = cpu
#     print('CPU:', max_cpu)

#     from jessetk2.RefineGlyph import Refine
#     r = Refine(dna_file, start_date, finish_date, eliminate, max_cpu)
#     r.run()


@cli.command()
@click.argument('start_date', required=True, type=str)
@click.argument('finish_date', required=True, type=str)
@click.argument('iterations', required=False, type=int)
@click.argument('width', required=False, type=int)
@click.option(
    '--cpu', default=0, show_default=True,
    help='The number of CPU cores that Jesse is allowed to use. If set to 0, it will use as many as is available on your machine.')
def random(start_date: str, finish_date: str, iterations: int, width: int, cpu: int) -> None:
    """
                                random walk backtest w/ threading.
                                Enter period "YYYY-MM-DD" "YYYY-MM-DD
                                Number of tests to perform  eg. 40
                                Sample width in days        eg. 30"
                                Thread counts to use        eg. 4
    """

    os.chdir(os.getcwd())
    validate_cwd()
    validateconfig()
    makedirs()

    if not iterations or iterations < 0:
        iterations = 32
        print(f'Iterations not provided, falling back to {iterations} iters!')
    if not width:
        width = 60
        print(
            f'Window width not provided, falling back to {width} days window!')

    max_cpu = utils.cpu_info(cpu)

    from jessetk2.RandomWalkTh import RandomWalk
    rwth = RandomWalk(start_date, finish_date, iterations, width, max_cpu)
    rwth.run()


@cli.command()
@click.argument('start_date', required=False, type=str)
def import_routes(start_date: str) -> None:
    """
    Import-candles for pairs listed in routes.py
    Enter start date "YYYY-MM-DD"
    If no start date is specified, the system will default to two days earlier.
    It's useful for executing script in a cron job to download deltas on a regular basis.
    """
    from dateutil.parser import isoparse
    import datetime

    try:
        isoparse(start_date)
        sd = start_date
    except:
        sd = str(datetime.date.today() - datetime.timedelta(days=2))
        print('Falling-back to two days earlier. Given parameter:', start_date)

    # os.chdir(os.getcwd())
    # validate_cwd()
    # validateconfig()

    try:
        from jesse.config import config
        from jesse.modes import import_candles_mode
        from jesse.routes import router
        from jesse.services.db import database
        config['app']['trading_mode'] = 'import-candles'
    except Exception as e:
        print(e)
        print('Check your routes.py file or database settings in config.py')
        exit()

    print(f'Startdate: {sd}')

    routes_list = router.routes

    if not routes_list or len(routes_list) < 1:
        print('Check your routes.py file!')
        exit()

    for t in routes_list:
        pair = t.symbol
        exchange = t.exchange
        print(f'Importing {exchange} {pair}')

        try:
            import_candles_mode.run('', exchange, pair, sd, running_via_dashboard=False)
        except KeyboardInterrupt:
            print('Terminated!')
            database.close_connection()
            sys.exit()
        except:
            print(f'Import error, skipping {exchange} {pair}')

    database.close_connection()


@cli.command()
@click.argument('dna_file', required=True, type=str)
@click.argument('start_date', required=True, type=str)
@click.argument('finish_date', required=True, type=str)
@click.argument('iterations', required=False, type=int)
@click.argument('width', required=False, type=int)
@click.option(
    '--cpu', default=0, show_default=True,
    help='The number of CPU cores that Jesse is allowed to use. If set to 0, it will use as many as is available on your machine.')
def randomrefine(dna_file: str, start_date: str, finish_date: str, iterations: int, width: int, cpu: int) -> None:
    """
                                random walk backtest w/ threading.
                                Enter period "YYYY-MM-DD" "YYYY-MM-DD
                                Number of tests to perform  eg. 40
                                Sample width in days        eg. 30
                                Thread counts to use        eg. 4
    """
    print('Not implemented yet!')
    exit()

    # os.chdir(os.getcwd())
    # validate_cwd()
    # validateconfig()
    # makedirs()

    # from jessetk2.Vars import datadir
    # os.makedirs(f'./{datadir}/results', exist_ok=True)

    # if cpu > cpu_count():
    #     raise ValueError(
    #         f'Entered cpu cores number is more than available on this machine which is {cpu_count()}')
    # elif cpu == 0:
    #     max_cpu = cpu_count()
    # else:
    #     max_cpu = cpu

    # print('Cpu count:', cpu_count(), 'Used:', max_cpu)

    # # if not eliminate:
    # #     eliminate = False
    # eliminate = False
    # from jessetk2.refine import refine
    # r = refine(dna_file, start_date, finish_date, eliminate)
    # r.run(dna_file, start_date, finish_date)

    # if not iterations or iterations < 0:
    #     iterations = 32
    #     print(f'Iterations not provided, falling back to {iterations} iters!')
    # if not width:
    #     width = 40
    #     print(
    #         f'Window width not provided, falling back to {width} days window!')

    # if cpu > cpu_count():
    #     raise ValueError(
    #         f'Entered cpu cores number is more than available on this machine which is {cpu_count()}')
    # elif cpu == 0:
    #     max_cpu = cpu_count()
    # else:
    #     max_cpu = cpu

    # print('Cpu count:', cpu_count(), 'Used:', max_cpu)
    # from jessetk2.RandomRefine import RandomRefine
    # from jessetk2.RandomWalkTh import RandomWalk

    # rrefine = RandomRefine(dna_file, start_date, finish_date, False)
    # rwth = RandomWalk(start_date, finish_date, iterations, width, max_cpu)
    # rwth.run()
    # #

    # rrefine.import_dnas()
    # rrefine.routes_template = utils.read_file('routes.py')

    # results = []
    # start = timer()
    # print_initial_msg()
    # for index, dnac in enumerate(rrefine.dnas, start=1):
    #     # Inject dna to routes.py
    #     utils.make_routes(rrefine.routes_template,
    #                       rrefine.anchor, dna_code=dnac[0])

    #     # Run jesse backtest and grab console output
    #     console_output = utils.run_test(start_date, finish_date)

    #     # Scrape console output and return metrics as a dict
    #     metric = utils.get_metrics3(console_output)

    #     if metric not in results:
    #         results.append(deepcopy(metric))
    #     # f.write(str(metric) + '\n')  # Logging disabled
    #     # f.flush()
    #     sorted_results_prelist = sorted(
    #         results, key=lambda x: float(x['sharpe']), reverse=True)
    #     rrefine.sorted_results = []

    #     if rrefine.eliminate:
    #         for r in sorted_results_prelist:
    #             if float(r['sharpe']) > 0:
    #                 rrefine.sorted_results.append(r)
    #     else:
    #         rrefine.sorted_results = sorted_results_prelist

    #     clear_console()

    #     eta = ((timer() - start) / index) * (rrefine.n_of_dnas - index)
    #     eta_formatted = strftime("%H:%M:%S", gmtime(eta))
    #     print(
    #         f'{index}/{rrefine.n_of_dnas}\teta: {eta_formatted} | {rrefine.pair} '
    #         f'| {rrefine.timeframe} | {rrefine.start_date} -> {rrefine.finish_date}')

    #     rrefine.print_tops_formatted()

    # utils.write_file('routes.py', rrefine.routes_template)  # Restore routes.py

    # if rrefine.eliminate:
    #     rrefine.save_dnas(rrefine.sorted_results, dna_file)
    # else:
    #     rrefine.save_dnas(rrefine.sorted_results)

    # utils.create_csv_report(rrefine.sorted_results,
    #                         rrefine.report_file_name, refine_file_header)


@cli.command()
@click.argument('dna_file', required=True, type=str)
@click.argument('start_date', required=True, type=str)
@click.argument('finish_date', required=True, type=str)
@click.argument('iterations', required=False, type=int)
@click.argument('width', required=False, type=int)
def randomsg(dna_file, start_date: str, finish_date: str, iterations: int, width: int) -> None:
    """
    random walk backtest w/ elimination
                                Enter period "YYYY-MM-DD" "YYYY-MM-DD
                                number of tests to perform  eg. 40
                                sample width in days        eg. 30"
    """

    os.chdir(os.getcwd())
    validate_cwd()
    validateconfig()
    makedirs()

    if start_date is None or finish_date is None:
        print('Enter dates!')
        exit()

    if not iterations or iterations < 0:
        iterations = 25
        print('Iterations not provided, falling back to 30 iters!')
    if not width:
        width = 50
        print(
            f'Window width not provided, falling back to {width} days window!')

    from jessetk2.randomwalk import RandomWalk
    random_walk = RandomWalk(start_date, finish_date, iterations, width)
    from jessetk2.refine import refine

    print_initial_msg()
    # start = timer()
    results = []
    for _ in range(1, iterations + 1):
        # Create a random period between given period
        rand_period_start, rand_period_finish = random_walk.make_random_period()

        r = refine(dna_file, rand_period_start,
                   rand_period_finish, eliminate=True)
        #      v ?
        r.run(dna_file, rand_period_start, rand_period_finish)

        if len(r.sorted_results) <= 5:
            print('Target reached, exiting...')
            break


@cli.command()
@click.argument('exchange', required=True, type=str)
@click.argument('symbol', required=True, type=str)
@click.argument('start_date', required=True, type=str)
@click.argument('data_type', required=False, default='klines', type=str)
@click.option(
    '--workers', default=4, show_default=True,
    help='The number of workers to run simultaneously. You can use cpu thread count or x2 or more.')
def bulkdry(exchange: str, symbol: str, start_date: str, data_type: str, workers: int) -> None:
    """
    DRY RUN
    Bulk download Binance candles as csv files. It does not save them to db.

    Enter EXCHANGE SYMBOL START_DATE { Optional: --workers n}

    jesse-tk bulkdry Binance btc-usdt 2020-01-01
    jesse-tk bulkdry 'Binance Futures' btc-usdt 2020-01-01

    You can use spot or futures keywords instead of full exchange name.

    jesse-tk bulkdry spot btc-usdt 2020-01-01
    jesse-tk bulkdry futures eth-usdt 2017-05-01 --workers 16
    """

    import arrow
    from dateutil import parser
    from jessetk2.Bulk import Bulk, get_days, get_months

    os.chdir(os.getcwd())
    validate_cwd()
    # validateconfig()

    try:
        start = parser.parse(start_date)
    except ValueError:
        print(f'Invalid start date: {start_date}')
        exit()

    symbol = symbol.upper()

    workers = max(workers, 8)

    try:
        sym = symbol.replace('-', '')
    except:
        print(f'Invalid symbol: {symbol}, format: BTC-USDT')
        exit()

    end = arrow.utcnow().floor('month').shift(months=-1)

    if exchange in {'binance', 'spot'}:
        exchange = 'Binance'
        market_type = 'spot'
        if data_type not in {'aggTrades', 'klines', 'trades'}:
            print(f'Invalid data type: {data_type}')
            print('Valid data types: aggTrades, klines, trades')
            exit()
        margin_type = None
    elif exchange in {'binance futures', 'futures'}:
        exchange = 'Binance Futures'
        market_type = 'futures'
        if data_type not in {'aggTrades', 'indexPriceKlines', 'klines', 'markPriceKlines',  'premiumIndexKlines', 'trades'}:
            print(f'Invalid data type: {data_type}')
            print('Valid data types: aggTrades, indexPriceKlines, klines, markPriceKlines, premiumIndexKlines, trades')
            exit()
        margin_type = 'um'
    else:
        print('Invalid market type! Enter: binance, binance futures, spot or futures')
        exit()

    # print start and end variables in color
    print(f'\x1b[36mStart: {start}\x1b[0m')
    print(f'\x1b[36mEnd: {end}\x1b[0m')

    b = Bulk(start=start, end=end, exchange=exchange, symbol=symbol,
             market_type=market_type, margin_type=margin_type, data_type=data_type, tf='1m', worker_count=workers)

    b.run()

    print('Completed in', round(timer() - b.timer_start), 'seconds.')


@cli.command()
@click.argument('exchange', required=True, type=str)
@click.argument('symbol', required=True, type=str)
@click.argument('start_date', required=True, type=str)
@click.option(
    '--workers', default=4, show_default=True,
    help='The number of workers to run simultaneously. You can use cpu thread count or x2.')
def bulk(exchange: str, symbol: str, start_date: str, workers: int) -> None:
    """
    Bulk download Binance candles
    Enter EXCHANGE SYMBOL START_DATE { Optional: --workers n}

    jesse-tk bulk Binance btc-usdt 2020-01-01
    jesse-tk bulk 'Binance Futures' btc-usdt 2020-01-01

    You can use spot or futures keywords instead of full exchange name.

    jesse-tk bulk spot btc-usdt 2020-01-01
    jesse-tk bulk futures eth-usdt 2017-05-01 --workers 8
    """

    import arrow
    from dateutil import parser
    from jessetk2.BulkJesse import BulkJesse

    os.chdir(os.getcwd())
    validate_cwd()
    # validateconfig()
    exchange = exchange.lower()

    try:
        start = parser.parse(start_date)
    except ValueError:
        print(f'Invalid start date: {start_date}')
        exit()

    workers = max(workers, 64)

    symbol = symbol.upper()

    try:
        sym = symbol.replace('-', '')
    except:
        print(f'Invalid symbol: {symbol}, format: BTC-USDT')
        exit()

    end = arrow.utcnow().floor('month').shift(months=-1)

    if exchange in ['binance', 'spot']:
        exchange = 'Binance'
        market_type = 'spot'
        margin_type = None
    elif exchange in ['binance futures', 'futures']:
        exchange = 'Binance Futures'
        market_type = 'futures'
        margin_type = 'um'
    else:
        print('Invalid market type! Enter: binance, binance futures, spot or futures')
        exit()

    print(f'\x1b[36mStart: {start}  {end}\x1b[0m')

    bb = BulkJesse(start=start, end=end, exchange=exchange,
                   symbol=symbol, market_type=market_type, tf='1m')

    bb.run()

    print('Completed in', round(timer() - bb.timer_start), 'seconds.')
# /////


@cli.command()
@click.argument('exchange', required=True, type=str)
@click.argument('start_date', required=True, type=str)
@click.option(
    '--workers', default=2, show_default=True,
    help='The number of workers to run simultaneously.')
@click.option('--all/--list', default=False, help="Get pairs list from api or pairs from file.")
def bulkpairs(exchange: str, start_date: str, workers: int, all) -> None:
    """
    Bulk download ALL! Binance Futures candles to Jesse DB.
    Enter EXCHANGE START_DATE { Optional: --workers n}

    jesse-tk bulkpairs 'Binance Futures' 2020-01-01
    jesse-tk bulkpairs futures 2017-05-01 --workers 8
    """

    exchange_data = {'binance': {'exchange': 'Binance', 'market_type': 'spot', 'margin_type': None},
                     'spot': {'exchange': 'Binance', 'market_type': 'spot', 'margin_type': None},
                     'binance futures': {'exchange': 'Binance Futures', 'market_type': 'futures', 'margin_type': 'um'},
                     'futures': {'exchange': 'Binance Futures', 'market_type': 'futures', 'margin_type': 'um'}}

    import arrow
    from dateutil import parser
    from jessetk2.BulkJesse import BulkJesse

    os.chdir(os.getcwd())
    validate_cwd()
    # validateconfig()
    exchange = exchange.lower()

    try:
        start = parser.parse(start_date)
    except ValueError:
        print(f'Invalid start date: {start_date}')
        exit()

    workers = max(workers, 2)

    end = arrow.utcnow().floor('month').shift(months=-1)
    
    print(exchange_data[exchange], exchange_data[exchange]['market_type'])

    if exchange in exchange_data:
        exchange_name = exchange_data[exchange]['exchange']
        market_type = exchange_data[exchange]['market_type']
        margin_type = exchange_data[exchange]['margin_type']

        if all:
            # Get pairs list from api
            from jessetk2.utils import get_symbols_list, avail_pairs
            pairs_list = get_symbols_list(exchange_name)
            db_symbols = avail_pairs(start_date, exchange_name)

            print(f"There's {len(pairs_list)} available pairs in {exchange_name}:")
            print(pairs_list)
            print(f"There's {len(db_symbols)} available pairs in candle database at: {start_date}")
        elif exchange_data[exchange]['market_type'] == 'spot':
            try:
                import pairs
                pairs_list = pairs.binance_spot_pairs
            except ImportError:
                print('Pairs file not found in project folder, loading default pairs list.')
                import jessetk2.pairs
                pairs_list = jessetk2.pairs.binance_spot_pairs
            except:
                print('Can not import pairs!')
                exit()
        elif exchange_data[exchange]['market_type'] == 'futures':
            try:
                import pairs
                pairs_list = pairs.binance_perp_pairs
            except ImportError:
                print('Pairs file not found in project folder, loading default pairs list.')
                import jessetk2.pairs
                pairs_list = jessetk2.pairs.binance_perp_pairs
            except:
                print('Can not import pairs!')
                exit()
    else:
        print('Invalid market type! Enter: binance, binance futures, spot or futures')
        exit()

    sloMo = False
    debug = False

    print(f'\x1b[36mStart: {start}  {end}\x1b[0m')

    bb = BulkJesse(start=start, end=end, exchange=exchange_name,
                   symbol='BTC-USDT', market_type=market_type, tf='1m')

    today = arrow.utcnow().format('YYYY-MM-DD')

    for pair in pairs_list:
        print(f'Importing {exchange_name} {pair} {start_date} -> {today}')
        # sleep2(5)
        bb.symbol = pair

        try:
            bb.run()
        except KeyboardInterrupt:
            print('Terminated!')
            sys.exit()
        except Exception as e:
            print(f'Error: {e}')
            continue
    print('Completed in', round(timer() - bb.timer_start), 'seconds.')

# ***************

# --------------------------------------------------


@cli.command()
@click.argument('dna_file', required=True, type=str)
@click.argument('start_date', required=True, type=str)
@click.argument('finish_date', required=True, type=str)
def refinepairs(dna_file, start_date: str, finish_date: str) -> None:
    """
    backtest all pairs with candidate dnas. Enter full path to dnafile and enter period in "YYYY-MM-DD" "YYYY-MM-DD"
    """
    os.chdir(os.getcwd())
    validate_cwd()

    from jessetk2.refinepairs import run
    validateconfig()
    makedirs()
    run(dna_file, _start_date=start_date, _finish_date=finish_date)


@cli.command()
def score() -> None:
    """
    z
    """
    os.chdir(os.getcwd())
    validate_cwd()

    from jessetk2.score import run
    validateconfig()
    makedirs()
    run()



@cli.command()
@click.argument('start_date', required=True, type=str)
@click.argument('finish_date', required=True, type=str)
def testpairs(start_date: str, finish_date: str) -> None:
    """
    backtest all candidate pairs. Enter in "YYYY-MM-DD" "YYYY-MM-DD"
    """
    os.chdir(os.getcwd())
    validate_cwd()

    from jessetk2.testpairs import run
    validateconfig()
    makedirs()
    run(_start_date=start_date, _finish_date=finish_date)


@cli.command()
@click.argument('strategy', required=True, type=str)
@click.argument('pairs_file', required=True, type=str)
@click.argument('start_date', required=True, type=str)
@click.argument('finish_date', required=True, type=str)
@click.option(
    '--tf', default='1h', show_default=True, help='Timeframe (e.g., 1h, 4h, 1D)')
@click.option(
    '--exchange', default='Binance Perpetual Futures', show_default=True,
    help='Exchange name')
@click.option(
    '--hp', default='', show_default=True, help='Hyperparameters as JSON dict')
@click.option(
    '--cpu', default=1, show_default=True,
    help='Number of concurrent workers (0 = all cores)')
def pairscan(strategy: str, pairs_file: str, start_date: str, finish_date: str,
             tf: str, exchange: str, hp: str, cpu: int) -> None:
    """
    Scan multiple pairs with a strategy using parquet backtest.

    Example: jesse-tk2 pairscan MyStrategy jessetkdata/pairfiles/testpairs.py 2024-01-01 2024-06-30 --tf 4h --cpu 4
    """
    import json
    makedirs()

    hp_dict = None
    if hp:
        hp_dict = json.loads(hp)

    if cpu == 0:
        cpu = os.cpu_count() - 1
        if cpu == 0: 
            cpu = 1

    from jessetk2.PairScan import run
    run(strategy=strategy, pairs_file=pairs_file, start_date=start_date,
        finish_date=finish_date, timeframe=tf, exchange=exchange, hp=hp_dict, cpu=cpu)