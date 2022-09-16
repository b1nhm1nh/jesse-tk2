import base64
from multiprocessing import cpu_count
import os
from subprocess import PIPE, Popen
import pandas as pd
from jesse.helpers import convert_number
import jessetk2.vars
from jessetk2.vars import random_console_formatter, random_console_header1, random_console_header2
from dateutil.parser import isoparse
import psycopg2
import urllib
import json
from datetime import datetime, timedelta
from dateutil.tz import UTC

def add_days(date: str, days: int):
    """Add days to a ISO formatted date string"""
    return (datetime.strptime(date, '%Y-%m-%d') + timedelta(days=days)).strftime('%Y-%m-%d')

def sub_days(date: str, days: int):
    """Subtract days from a ISO formatted date string"""
    return (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=days)).strftime('%Y-%m-%d')

def get_symbols_list(exchange: str = 'Binance Futures', quote_asset: str = 'USDT') -> list:
    local_fn = f"{exchange.replace(' ', '')}ExchangeInfo.json"
    urls = {'Binance': 'https://api.binance.com/api/v1/exchangeInfo', 'Binance Futures': 'https://fapi.binance.com/fapi/v1/exchangeInfo'}
    symbols_list = []

    try:
        with urllib.request.urlopen(urls[exchange]) as url:
            data = json.loads(url.read().decode())

        # save url to file
        if int(data['serverTime']):
            try:
                with open(local_fn, 'w') as f:
                    json.dump(data, f)
            except:
                print(f"Failed to save {local_fn}")

    except Exception as e:
        print(f"Error while fetching data from {exchange}. {e}")

        try:
            with open(local_fn) as f:
                data = json.load(f)

            print(f"Using cached local data from {datetime.utcfromtimestamp(data['serverTime'] / 1000).strftime('%Y-%m-%d %H:%M')}")
        except Exception as e:
            print(f"Error while loading local api data for {exchange}. {e}")
            return None

    symbols = []

    for sym in data['symbols']:
        blvt = False

        # Check if symbol is a leveraged token (UP/DOWN)

        try:
            blvt  = 'LEVERAGED' in sym['permissions']
        except:
            pass        

        if sym['quoteAsset'] == quote_asset and not blvt and sym['status'] != 'BREAK':
            # print(f"{sym['baseAsset']}-{quote_asset}")
            symbols.append(f"{sym['baseAsset']}-{quote_asset}")

    return symbols or None

def avail_pairs(start_date: str = '2021-08-01', exchange: str = 'Binance Futures') -> list:
    from jesse.config import config
    symbols_list = None
    date = isoparse(f'{start_date}T00:00:00+00:00').astimezone(UTC)
    epoch = int(date.timestamp() * 1000)

    db_name = config['env']['databases']['postgres_name']
    db_user = config['env']['databases']['postgres_username']
    db_pass = config['env']['databases']['postgres_password']
    db_host = config['env']['databases']['postgres_host']
    db_port = config['env']['databases']['postgres_port']

    try:
        conn = psycopg2.connect(database=db_name, user=db_user, password=db_pass, host=db_host, port=db_port)

        cursor = conn.cursor()
        query = f"select symbol from public.candle where exchange = '{exchange}' and timestamp = {epoch} group by symbol;"
        cursor.execute(query)
        symbols = cursor.fetchall()
        symbols_list = [x[0] for x in symbols]

    except (Exception, psycopg2.Error) as error:
        print("Error while fetching data from PostgreSQL", error)
        return []

    finally:
        if conn:
            cursor.close()
            conn.close()
        return symbols_list or []

def hp_to_seq(hp):
    longest_param = 0

    for v in hp.values():
        if len(str(v)) > longest_param:
            longest_param = len(str(v))
            
    hash = ''.join([f'{value:0>{longest_param}}' for key, value in hp.items()])
    return f"{hash}{longest_param}"


def decode_seq(seq):
    # Get the sequence width from the last char
    width = int(seq[-1])
    # Remove the width from the sequence
    seq = seq[:-1]
    return [seq[i:i+width] for i in range(0, len(seq), width)]


def hp_to_dna(strategy_hp, hps):    # TODO - make it work with floats too
    """Returns DNA code from HP parameters
    example input: {'ott_len': 3, 'ott_percent': 420, 'stop_loss': 329, 'risk_reward': 19, 'chop_rsi_len': 27, 'chop_bandwidth': 21}
    example output: *Og2O+

    Args:
        strategy_hp (StrategyClass.hyperparameters(None): Hyperparameters defined in StrategyClass
        hps (Dict): Values to be converted to DNA

    Returns:
        str: encoded DNA string
    """
    return ''.join(
        chr(
            int(
                round(
                    convert_number(
                        param['max'], param['min'], 119, 40, hps[dec]
                    )
                )
            )
        )
        for dec, param in zip(hps, strategy_hp)
    )


def cpu_info(cpu):
    if cpu > cpu_count():
        raise ValueError(
            f'Entered cpu cores number is more than available on this machine which is {cpu_count()}')
    elif cpu == 0:
        max_cpu = cpu_count()
    else:
        max_cpu = cpu
    print('Cpu count:', cpu_count(), 'In use:', max_cpu)
    return max_cpu


def encode_base32(s):
    s_bytes = s.encode('ascii')
    base32_bytes = base64.b32encode(s_bytes)
    return base32_bytes.decode('ascii')


def decode_base32(b):
    base32_bytes = b.encode('ascii')
    try:
        message_bytes = base64.b32decode(base32_bytes)
    except:
        print(b)
        exit()
    return message_bytes.decode('ascii')


def clear_console(): return os.system(
    'cls' if os.name in ('nt', 'dos') else 'clear')


def run_test(start_date, finish_date):
    process = Popen(['jesse', 'backtest', start_date,
                     finish_date], stdout=PIPE)
    (output, err) = process.communicate()
    exit_code = process.wait()
    return output.decode('utf-8')


def make_routes(template, anchor, dna_code):
    if anchor not in template:
        os.system('color')
        print('\nReplace the dna strings in routes.py with anchors. eg:\n')
        print(
            f"""(\033[32m'Bitfinex', 'BTC-USD', '2h', 'myStra', '{anchor}'\033[0m),\n""")
        exit()

    write_file('routes.py', template.replace(
        "'" + anchor + "'", repr(dna_code)))


def split(line):
    ll = line.split(' ')
    r = ll[len(ll) - 1].replace('%', '')
    r = r.replace(')', '')
    r = r.replace('(', '')
    return r.replace(',', '')

def split_dna_string(line):
    ll = line.split(' ')
    return ll[len(ll) - 1]
    


def split_n_of_longs_shorts(line):
    ll = line.split(' ')
    shorts = ll[len(ll) - 1].replace('%', '')
    longs = ll[len(ll) - 3].replace('%', '')
    return int(longs), int(shorts)


def split_dates(line):
    return line.replace(' ', '').split('|')[-1].split('=>')  # Lazy man's reg^x


def split_estfd(line):
    # Split Exchange, Symbol, Timeframe, Strategy,
    # DNA while keeping spaces in exchange name
    return [x.strip() for x in line.split('|')]


def print_tops_formatted(frmt, header1, header2, tr):
    print('\x1b[6;34;40m')
    print(
        frmt.format(*header1))
    print(
        frmt.format(*header2))
    print('\x1b[0m')

    for r in tr:
        print(
            frmt.format(
                r['dna'], r['total_trades'],
                r['n_of_longs'], r['n_of_shorts'],
                r['total_profit'], r['max_dd'],
                r['annual_return'], r['win_rate'],
                r['serenity'], r['sharpe'], r['calmar'],
                r['win_strk'], r['lose_strk'],
                r['largest_win'], r['largest_lose'],
                r['n_of_wins'], r['n_of_loses'],
                r['paid_fees'], r['market_change']))


def print_random_header():
    print(
        random_console_formatter.format(*random_console_header1))
    print(
        random_console_formatter.format(*random_console_header2))


def print_random_tops(sr, top_n):
    for r in sr[:top_n]:
        print(
            random_console_formatter.format(
                r['start_date'],
                r['finish_date'],
                r['total_trades'],
                r['n_of_longs'],
                r['n_of_shorts'],
                r['total_profit'],
                r['max_dd'],
                r['annual_return'],
                r['win_rate'],
                r['serenity'],
                r['sharpe'],
                r['calmar'],
                r['win_strk'],
                r['lose_strk'],
                r['largest_win'],
                r['largest_lose'],
                r['n_of_wins'],
                r['n_of_loses'],
                r['paid_fees'],
                r['market_change']))


def print_tops_generic(frmt, header1, header2, tr):
    print(
        frmt.format(*header1))
    print(
        frmt.format(*header2))

    for r in tr:
        print(frmt.format(*tr))


def create_csv_report(sorted_results, filename, header):
    from jessetk2.vars import csvd
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(str(header).replace('[', '').replace(']', '').replace("'", "").replace(',',
                                                                                       csvd) + '\n')
        #repr
        for srl in sorted_results:
            f.write(f"{srl['symbol']}{csvd}{srl['tf']}{csvd}" + (srl['dna']) +
                    f"{csvd}{srl['start_date']}{csvd}"
                    f"{srl['finish_date']}{csvd}"
                    f"{srl['total_trades']}{csvd}"
                    f"{srl['n_of_longs']}{csvd}"
                    f"{srl['n_of_shorts']}{csvd}"
                    f"{srl['total_profit']}{csvd}"
                    f"{srl['max_dd']}{csvd}"
                    f"{srl['annual_return']}{csvd}"
                    f"{srl['win_rate']}{csvd}"
                    f"{srl['serenity']}{csvd}"
                    f"{srl['sharpe']}{csvd}"
                    f"{srl['calmar']}{csvd}"
                    f"{srl['win_strk']}{csvd}"
                    f"{srl['lose_strk']}{csvd}"
                    f"{srl['largest_win']}{csvd}"
                    f"{srl['largest_lose']}{csvd}"
                    f"{srl['n_of_wins']}{csvd}"
                    f"{srl['n_of_loses']}{csvd}"
                    f"{srl['paid_fees']}{csvd}"
                    f"{srl['market_change']}\n")


def get_metrics3(console_output) -> dict:
    metrics = jessetk2.vars.Metrics
    lines = console_output.splitlines()

    for index, line in enumerate(lines):
        if 'Aborted!' in line:
            print(console_output)
            print(
                "Aborted! error. Possibly pickle database is corrupt. Delete temp/ folder to fix.")
            exit(1)

        if 'CandleNotFoundInDatabase' in line:
            print(console_output)
            return metrics

        if 'Uncaught Exception' in line:
            print(console_output)
            if 'must be within the range' in line:
                print('Check DNA String in routes file!')
            exit(1)

        if 'No trades were made' in line:
            return metrics
        
        if 'InsufficientMargin' in line:
            print(console_output)
            return metrics

        if 'starting-ending date' in line:
            metrics['start_date'], metrics['finish_date'] = split_dates(line)

        if 'exchange' in line and 'symbol' in line and 'timeframe' in line:
            metrics['exchange'], metrics['symbol'], metrics['tf'], metrics['strategy'], metrics['dna'] = split_estfd(
                lines[index + 2])

        if 'Total Closed Trades' in line:
            metrics['total_trades'] = int(split(line))

        if 'Total Net Profit' in line:
            metrics['total_profit'] = round(float(split(line)), 2)

        if 'Max Drawdown' in line:
            metrics['max_dd'] = round(float(split(line)), 2)

        if 'Total Paid Fees' in line:
            metrics['paid_fees'] = round(float(split(line)), 2)

        if 'Annual Return' in line:
            metrics['annual_return'] = round(float(split(line)), 2)

        if 'Percent Profitable' in line:
            metrics['win_rate'] = int(split(line))

        if 'Serenity Index' in line:
            metrics['serenity'] = round(float(split(line)), 2)

        if 'Sharpe Ratio' in line:
            metrics['sharpe'] = round(float(split(line)), 2)

        if 'Calmar Ratio' in line:
            metrics['calmar'] = round(float(split(line)), 2)

        if 'Sortino Ratio' in line:
            metrics['sortino'] = round(float(split(line)), 2)

        if 'Smart Sharpe' in line:
            metrics['smart_sharpe'] = round(float(split(line)), 2)

        if 'Smart Sortino' in line:
            metrics['smart_sortino'] = round(float(split(line)), 2)

        if 'Winning Streak' in line:
            metrics['win_strk'] = int(split(line))

        if 'Losing Streak' in line:
            metrics['lose_strk'] = int(split(line))

        if 'Longs | Shorts' in line:
            metrics['n_of_longs'], metrics['n_of_shorts'] = split_n_of_longs_shorts(
                line)

        if 'Largest Winning Trade' in line:
            metrics['largest_win'] = round(float(split(line)), 2)

        if 'Largest Losing Trade' in line:
            metrics['largest_lose'] = round(float(split(line)), 2)

        if 'Total Winning Trades' in line:
            metrics['n_of_wins'] = int(split(line))

        if 'Total Losing Trades' in line:
            metrics['n_of_loses'] = int(split(line))

        if 'Market Change' in line:
            metrics['market_change'] = round(float(split(line)), 2)
            
        if 'Dna String:' in line:
            metrics['dna'] = split_dna_string(line)
        
        if 'Sequential Hps' in line:
            metrics['seq_hps'] = split(line)

        if 'Report Prefix' in line:
            metrics['dna'] = line.split("|")[1]

        if 'JSON Metrics' in line:
            metrics['json_metrics'] = line.split("|")[1]
    return metrics


def write_file(fn, body):
    remove_file(fn)
    with open(fn, 'w', encoding='utf-8') as f:
        f.write(body)
        f.flush()
        os.fsync(f.fileno())


def read_file(fn):
    with open(fn, 'r', encoding='utf-8') as f:
        return f.read()


def remove_file(fn):
    if os.path.exists(fn):
        try:
            os.remove(fn)
        except:
            print(f'Failed to remove file {fn}')
            exit()

def read_csv_file(filename):
    """
    Read csv with headers using pandas
    :param filename:
    """
    df = pd.read_csv(filename, sep='\t') #, lineterminator='\r')
    return df

def import_dnas(filename, max_dnas = 1000):
    """
    Import DNA from file
    :param
    """
    dnas = read_csv_file(filename)
    #replace header
    columns = []
    for str in dnas.columns:
        str = str.replace('training_log','tn')
        str = str.replace('testing_log','tt')
        str = str.replace('parameters','p')
        columns.append(str)
    dnas.columns = columns
    #remove dupicate dnas
    dnas.drop_duplicates(subset=['dna'], keep='first', inplace=True)
    #remove dnas with negative pnl total
    dnas.drop(dnas[dnas['tn.net_profit'] < 0].index, inplace = True)
    dnas.drop(dnas[dnas['tt.net_profit'] < 0].index, inplace = True)

    # top_ss2 = dnas.sort_values(by=['tt.smart_sortino','tn.smart_sortino'], ascending=False)
    # print(top_ss2[header].head(20))
    top_sr = dnas.sort_values(by=['tn.sharpe_ratio','tt.sharpe_ratio'], ascending=False)
    # print(top_sr[header].head(20))

    return top_sr.head(max_dnas)

def import_dnas3(filename, max_dnas = 1000):
    """
    Import DNA from file
    :param
    """
    dnas = read_csv_file(filename)
    #replace header
    columns = []
    for str in dnas.columns:
        str = str.replace(' Max.DD','Max.DD')        
        str = str.replace(' Dna','dna')
        str = str.replace(' Sharpe','Sharpe')
        str = str.replace(' Total Net Profit','Total Net Profit')
    #     str = str.replace('parameters','p')
        columns.append(str)
    dnas.columns = columns
    #remove dupicate dnas
    print(dnas.columns)
    print(dnas.head(10))

    dnas.drop_duplicates(subset=['dna'], keep='first', inplace=True)
    #remove dnas with negative pnl total
    dnas.drop(dnas[dnas['Total Net Profit'] < 0].index, inplace = True)
    # dnas.drop(dnas[dnas['Max.DD'] < -25].index, inplace = True)

    # top_ss2 = dnas.sort_values(by=['tt.smart_sortino','tn.smart_sortino'], ascending=False)
    # print(top_ss2[header].head(20))
    top_sr = dnas.sort_values(by=['Sharpe'], ascending=False)
    # print(top_sr[header].head(20))

    return top_sr.head(max_dnas)

def make_route(filename: str, out_filename:str, exchange: str, symbol: str, timeframe: str, strategy: str):
    # Read in the file
    with open(filename, 'r') as file:
        filedata = file.read()

    # Replace the target string
    filedata = filedata.replace('EXCHANGE',  exchange)
    filedata = filedata.replace('SYMBOL',    symbol)
    filedata = filedata.replace('TIMEFRAME', timeframe)
    filedata = filedata.replace('STRATEGY',  strategy)

        # Write the file out again
    with open(out_filename, 'w') as file:
        file.write(filedata)