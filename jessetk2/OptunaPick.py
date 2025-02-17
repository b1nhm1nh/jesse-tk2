import sys
import click
import optuna
import statistics
import json
from jessetk2.utils import hp_to_seq
import csv
import jesse.helpers as jh

try:
    from config import config
except ImportError:
    print('Check your config.py file or project folder structure. You need legacy jesse cli!')
    sys.exit(1)

try:
    from routes import routes as routes_cli
except ImportError:
    print("Error: routes.py not found")
    sys.exit(1)



class OptunaPick:
    def __init__(self, t1=0.001, t2=0, t3=-90):
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3

        try:
            self.db_host = config['databases']['optuna_db_host']
            self.db_port = config['databases']['optuna_db_port']
            self.db_name = config['databases']['optuna_db']
            self.db_user = config['databases']['optuna_user']
            self.db_password = config['databases']['optuna_password']
        except:
            print(
                'Check your config.py file for optuna database settings! example configuration:')
            print("""
                'databases': {
                'postgres_host': '127.0.0.1',
                'postgres_name': 'jesse_db',
                'postgres_port': 5432,
                'postgres_username': 'jesse_user',
                'postgres_password': 'password',
                
                'optuna_db_host': '127.0.0.1',
                'optuna_db_port': 5432,
                'optuna_db': 'optuna_db',
                'optuna_user': 'optuna_user',
                'optuna_password': 'password',
                },
            """)
            exit()

        self.storage = f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}/{self.db_name}"

    def dump_best_parameters(self):
        study_summaries = optuna.study.get_all_study_summaries(storage=self.storage)
        # Sort study_summaries by datetime_start
        studies_sorted = sorted(study_summaries, key=lambda x: x._study_id)

        print(f"{'-'*10} {'-'*8} {'-'*26} {'-'*64}")
        print(f"{'Study ID':<10} {'Trials':<8} {'Datetime':<26} {'Study Name':<64}")
        print(f"{'-'*10} {'-'*8} {'-'*26} {'-'*64}")

        studies_dict = {}

        for ss in studies_sorted:
            studies_dict[ss._study_id] = ss.study_name
            print(
                f"{ss._study_id:<10} {ss.n_trials:<8} {str(ss.datetime_start):<26} {ss.study_name:<64}")

        value = click.prompt('Pick a study', type=int)

        try:
            study_name = studies_dict[value]
            study = optuna.load_study(study_name=study_name, storage=self.storage)
        except:
            print('Study not found!')
            exit()

        print(value, study_name)
        print("Number of finished trials: ", len(study.trials))

        # sorted(study.best_trials, key=lambda t: t.values)
        trials = study.trials
        results = []
        parameter_list = []  # to eliminate redundant trials with same parameters
        candidates = {}

        exchange = routes_cli[0][0]
        symbol = routes_cli[0][1]
        timeframe = routes_cli[0][2]
        strategy_name = routes_cli[0][3]
        StrategyClass = jh.get_strategy_class(strategy_name)
        strategy = StrategyClass()

        for trial in trials:
            total_profit = max_mr = None
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue

            # Check each trial values
            # if any(v < 0 for v in trial.values):  # 1
            #     continue

            if (not trial.user_attrs['trades1']) or trial.user_attrs['trades1'] < 5:
                continue

            # Make this part customizable for each strategy
            # It's hardcoded for k series for now.

            # Total profit
            if trial.values[0] < self.t1:
                continue

            # Max DD
            if trial.values[1] > self.t2:
                continue
            
            # try:
            #     if trial.user_attrs['max_dd1'] < -self.t3:
            #         continue
            # except:
            #     pass

            total_profit = trial.values[0]
            metric2 = trial.values[1]

            # Statistics test are useful for some strategies!
            # mean_value = round(statistics.mean((*trial.values, trial.user_attrs['sharpe3'])), 3)
            # std_dev = round(statistics.stdev((*trial.values, trial.user_attrs['sharpe3'])), 5)

            # {key : round(trial.params[key], 5) for key in trial.params}
            rounded_params = trial.params

            # Inject payload HP to route
            hp_new = {}

            # Sort hyperparameters as defined in the strategy
            for p in strategy.hyperparameters():
                hp_new[p['name']] = rounded_params[p['name']]

            rounded_params = hp_new

            # Remove duplicates
            # and mean_value > score_treshold and std_dev < std_dev_treshold:
            if trial.params not in parameter_list:
                hash = hp_to_seq(rounded_params)
                # candidates.append([hash, hp])
                candidates[hash] = rounded_params
                # print(type(trial.values), trial.values)

                # This is also hardcoded for k series for now.
                result_line = [
                    trial.number, f"'{hash}'",
                    *trial.values,
                    trial.user_attrs['max_dd1'],
                    trial.user_attrs['sharpe1'],
                    trial.user_attrs['trades1'],
                    trial.user_attrs['fees1'],
                    rounded_params]

                results.append(result_line)
                parameter_list.append(trial.params)

                # If parameters meet criteria, add to candidates
                # if mean_value > score_treshold and std_dev < std_dev_treshold and  trial.user_attrs['sharpe3'] > 2:

        results = sorted(results, key=lambda x: x[2], reverse=True)
        print(f"Picked {len(results)} trials")

        # field names
        fields = ['Trial #', 'Seq', 'Profit', 'insufMargin',
                  'Max DD', 'Sharpe', 'Trades', 'Fees', 'HP']

        with open(f'Results-{self.db_name}-{study_name.replace(" ", "-")}.csv', 'w') as f:
            # using csv.writer method from CSV package
            write = csv.writer(f, delimiter='\t', lineterminator='\n')

            write.writerow(fields)
            write.writerows(results)

        with open(f'SEQ-{self.db_name}-{study_name.replace(" ", "-")}.py', 'w') as f:
            f.write("hps = ")
            f.write(json.dumps(candidates, indent=1))

        print("Tada! Results saved to file.")
