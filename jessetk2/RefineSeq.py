import importlib
import os
import sys
import platform
from copy import deepcopy
from datetime import datetime
from subprocess import PIPE, Popen
from time import sleep, strftime, gmtime
from timeit import default_timer as timer
from jesse.routes import router


import jessetk2.Vars as Vars
import jessetk2.utils
from jessetk2 import utils
from jessetk2.Vars import datadir
from jessetk2.Vars import refine_file_header
import json
from millify import millify
from importlib.metadata import version
from routes import routes as routes_cli

class Refine:
    def __init__(self, hp_py_file, start_date, finish_date, eliminate, cpu, dd, mr, lpr, sharpe, profit, imcount, sortby='sharpe', full_reports=False):
        import signal
        signal.signal(signal.SIGINT, self.signal_handler)

        self.hp_py_file = hp_py_file
        self.start_date = start_date
        self.finish_date = finish_date
        self.cpu = cpu
        self.eliminate = eliminate
        self.dd = dd
        self.mr = mr
        self.lpr = lpr
        self.sharpe = sharpe
        self.profit = profit
        self.imcount = imcount
        self.sortby = sortby.replace('profit', 'total_profit')
        # Minimum is better for max lp rate, so we need to reverse the sort
        self.sort_reverse = sortby != 'lpr'
        self.fr = ' --full-reports' if full_reports else ''
        self.jessetkdir = datadir
        self.anchor = 'DNA!'
        self.sort_by = {'serenity': 12, 'sharpe': 13, 'calmar': 14}

        self.metrics = []

        self.n_of_iters = 0
        self.results = []
        self.sorted_results = []
        self.results_without_dna = []

        self.hps_module = None
        self.routes_template = None
        self.params = None
        self.n_of_params = None

        self.exchange = routes_cli[0][0]
        self.pair = symbol = routes_cli[0][1]
        self.timeframe = routes_cli[0][2]
        self.strategy = strategy_name = routes_cli[0][3]

        self.removesimilardnas = False

        self.ts = datetime.now().strftime("%Y%m%d %H%M%S")
        self.filename = f'RefineHp-{self.exchange}-{self.pair}-{self.timeframe}--{start_date}--{finish_date}'

        self.report_file_name = f'{self.jessetkdir}/results/{self.filename}--{self.ts}.csv'
        self.log_file_name = f'{self.jessetkdir}/logs/{self.filename}--{self.ts}.log'

        self.os = platform.system()
        self.interpreter = sys.executable

    def run(self):
        max_cpu = self.cpu
        processes = []
        commands = []
        results = []
        sorted_results = []
        iters_completed = 0
        self.import_dnas()
        iters = self.n_of_params
        self.n_of_iters = self.n_of_params
        index = 0  # TODO Reduce number of vars ...
        start = timer()
        print(f'OS: {self.os}, Interpreter: {self.interpreter}')

        while iters > 0:
            commands = []

            for _ in range(max_cpu):
                if iters > 0:
                    hps = self.params[index]
                    # hps = json.dumps(hps).replace('"', '%')
                    # print(f'parameters: {hps}')

                    commands.append(
                        f'jesse-tk2 backtest {self.start_date} {self.finish_date} --seq {hps}{self.fr}'
                        )

                    index += 1
                    iters -= 1
            # print(f'commands: {commands}')
            processes = [Popen(cmd, shell=True, stdout=PIPE) for cmd in commands]

            # wait for completion
            for p in processes:
                p.wait()

                # Get thread's console output
                (output, err) = p.communicate()
                # debug
                print(output.decode('utf-8'))

                try:
                    print(err.decode('utf-8'))
                except Exception as e:
                    print(e)
                    pass
                # exit()
                iters_completed += 1

                # Map console output to a dict
                metric = utils.get_metrics3(output.decode('utf-8'))
                metric['dna'] =  metric['seq_hps']

                # print('Metrics decoded', len(metric))

                if metric not in results:
                    results.append(deepcopy(metric))

                sorted_results_prelist = sorted(results, key=lambda x: float(x[self.sortby]), reverse=self.sort_reverse)
                # print(f'Sorted results: {sorted_results_prelist}')
                # print('Sorted results', len(sorted_results_prelist))

                # sleep(10)
                self.sorted_results = []

                if self.eliminate:
                    for r in sorted_results_prelist:
                        if float(r['sharpe']) > 0:
                            self.sorted_results.append(r)
                else:
                    self.sorted_results = sorted_results_prelist

                utils.clear_console()

                eta = ((timer() - start) / index) * (self.n_of_params - index)
                eta_formatted = strftime("%H:%M:%S", gmtime(eta))

                print(
                    f'{index}/{self.n_of_params}\teta: {eta_formatted} | {self.pair} |'
                    f' {self.timeframe} | {self.start_date} -> {self.finish_date} |'
                    f" Sort by {self.sortby} {'reversed' if self.sort_reverse else ''} |"
                    f" Filters: IM: {self.imcount} | MR% {self.mr}, DD% {self.dd}, LPR {self.lpr}, Sharpe {self.sharpe}, Profit: {self.profit} | Ver. {version('jesse-tk2')}")

                self.print_tops_formatted(n=30)

        # if self.eliminate:
        #     self.save_dnas(self.sorted_results, self.dna_py_file)
        # else:
        #     self.save_dnas(self.sorted_results)

        # self.save_seq(self.sorted_results)

        if 'spot' in self.exchange.lower():
            candidates = {
                r['dna']: r['dna']
                for r in self.sorted_results
                if r['max_dd'] > self.dd and r['sharpe'] > self.sharpe and r['total_profit'] > self.profit and r['insuff_margin_count'] <= self.imcount
                # if r['max_dd'] > self.dd and r['sharpe'] > self.sharpe and r['total_profit'] > self.profit and r['insuff_margin_count'] <= self.imcount
            }
        else:
            candidates = {
                r['dna']: r['dna']
                for r in self.sorted_results
                if r['max_dd'] > self.dd and r['max_margin_ratio'] < self.mr and r['lpr'] < self.lpr and r['sharpe'] > self.sharpe and r['total_profit'] > self.profit and r['insuff_margin_count'] <= self.imcount
                # if r['max_dd'] > self.dd and r['sharpe'] > self.sharpe and r['total_profit'] > self.profit and r['insuff_margin_count'] <= self.imcount
            }

        print(f'\n\nCandidates: {len(candidates)}')
        seq_fn = f'SEQ-{self.pair}-{self.strategy}-{self.start_date}-{self.finish_date}.py'

        with open(seq_fn, 'w') as f:
            f.write("hps = ")
            f.write(json.dumps(candidates, indent=1))

        with open('last_seq_fn', 'w') as f:
            f.write(seq_fn)

        utils.create_csv_report(self.sorted_results,
                                self.report_file_name, refine_file_header)


    def signal_handler(self, sig, frame):
        print('You pressed Ctrl+C!')
        sys.exit(0)

    def import_dnas(self):
        module_name = self.hp_py_file.replace('.\\', '').replace('.py', '')
        module_name = module_name.replace('/', '.').replace('.py', '')
        print(module_name)

        self.hps_module = importlib.import_module(module_name)
        importlib.reload(self.hps_module)
        self.params = [*self.hps_module.hps]  # self.hps_module.hps.keys()
        self.n_of_params = len(self.params)
        print(f'Imported {self.n_of_params} parameters...')
        # print('self.params', self.params)
        # sleep(5)
        
    # v TODO Move to utils
    def print_tops_formatted(self, sorted_results=None, n:int = 50):
        if sorted_results is None:
            sorted_results = self.sorted_results
            
        print(
            jessetk2.Vars.refine_console_formatter.format(*jessetk2.Vars.refine_console_header1))
        print(
            jessetk2.Vars.refine_console_formatter.format(*jessetk2.Vars.refine_console_header2))

        for r in sorted_results[:n]:
            p = r
            # Replace None with empty string
            for k, v in p.items():
                if v is None:
                    p[k] = ''

            # p = {}
            # # make a copy of r dict but round values if they are floats
            # for k, v in r.items():
            #     try:
            #         if type(v) is float and v > 999999:
            #             p[k] = millify(v, 2)
            #         elif type(v) is float and abs(v) > 999:
            #             p[k] = round(v)
            #         else:
            #             p[k] = v
            #     except:
            #         p[k] = v

            # for i in range(len(r)):
            #     if isinstance(r[i], float) and r[i] > 999999:
            #         p.append(millify(round(r[i]), 2))  # '{:.2f}'.format(r[i])
            #     # elif isinstance(r[i], float) and r[i] > 1000:
            #     #     p.append(round(r[i], 2))
            #     else:
            #         p.append(r[i])

            print(
                jessetk2.Vars.refine_console_formatter.format(
                    p['dna'],
                    p['total_trades'],
                    p['n_of_longs'],
                    p['n_of_shorts'],
                    p['total_profit'],
                    p['max_margin_ratio'],
                    p['pmr'],
                    p['lpr'],
                    p['insuff_margin_count'],
                    p['max_dd'],
                    p['annual_return'],
                    p['win_rate'],
                    p['serenity'],
                    p['sharpe'],
                    p['calmar'],
                    p['win_strk'],
                    p['lose_strk'],
                    p['largest_win'],
                    p['largest_lose'],
                    p['n_of_wins'],
                    p['n_of_loses'],
                    p['paid_fees'],
                    p['market_change'],
                    p['bench_vol']
                    ))

    # def print_tops_formatted(self):
    #     print(
    #         Vars.refine_console_formatter.format(*Vars.refine_console_header1))
    #     print(
    #         Vars.refine_console_formatter.format(*Vars.refine_console_header2))

    #     for r in self.sorted_results[:25]:
            
    #         p = {}
    #         # make a copy of r dict but round values if they are floats
    #         for k, v in r.items():
    #             if type(v) is float and v > 999999:
    #                 p[k] = millify(v, 2)
    #             elif type(v) is float and abs(v) > 999:
    #                 p[k] = round(v)
    #             else:
    #                 p[k] = v

    #         # for i in range(len(r)):
    #         #     if isinstance(r[i], float) and r[i] > 999999:
    #         #         p.append(millify(round(r[i]), 2))  # '{:.2f}'.format(r[i])
    #         #     # elif isinstance(r[i], float) and r[i] > 1000:
    #         #     #     p.append(round(r[i], 2))
    #         #     else:
    #         #         p.append(r[i])

    #         print(
    #             Vars.refine_console_formatter.format(
    #                 p['dna'],
    #                 p['total_trades'],
    #                 p['n_of_longs'],
    #                 p['n_of_shorts'],
    #                 p['total_profit'],
    #                 p['max_dd'],
    #                 p['annual_return'],
    #                 p['win_rate'],
    #                 p['serenity'],
    #                 p['sharpe'],
    #                 p['calmar'],
    #                 p['win_strk'],
    #                 p['lose_strk'],
    #                 p['largest_win'],
    #                 p['largest_lose'],
    #                 p['n_of_wins'],
    #                 p['n_of_loses'],
    #                 p['paid_fees'],
    #                 p['market_change']))

    def save_dnas(self, sorted_results, dna_fn=None):

        if not dna_fn:
            dna_fn = f'{self.jessetkdir}/dnafiles/{self.pair} {self.start_date} {self.finish_date}.py'

        jessetk2.utils.remove_file(dna_fn)

        with open(dna_fn, 'w', encoding='utf-8') as f:
            self.write_dna_file(f, sorted_results)

    def write_dna_file(self, f, sorted_results):
        f.write('dnas = [\n')

        for srr in sorted_results:
            for dnac in self.dnas:
                if srr['dna'] == dnac[0]:
                    f.write(str(dnac) + ',\n')

        f.write(']\n')
        f.flush()
        os.fsync(f.fileno())
