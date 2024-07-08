"""
Módulo responsavel pela execução de campanhas da ferramenta 

"""
# Importação de bibliotecas necessárias
try:
    import sys
    import os
    import argparse
    import logging
    import subprocess
    import shlex
    import datetime
    from logging.handlers import RotatingFileHandler
    from pathlib import Path
    import itertools
    import mlflow

except ImportError as error:
    print(error)
    print()
    print(" ")
    print()
    sys.exit(-1)

# Definindo constantes padrão
DEFAULT_VERBOSITY_LEVEL = logging.INFO
NUM_EPOCHS = 1000
TIME_FORMAT = '%Y-%m-%d_%H:%M:%S'
DEFAULT_CAMPAIGN = "demo"
PATH_LOG = 'logs'
PATH_DATASETS = 'datasets'
PATHS = [PATH_LOG]
Parâmetros = None
COMMAND = "pipenv run python main.py   "
COMMAND2 = "pipenv run python main.py -ml  "

datasets = ['datasets/kronodroid_emulador-balanced.csv', 'datasets/kronodroid_real_device-balanced.csv']

# Definindo campanhas disponíveis
campaigns_available = {
    'demo': {
        'input_dataset': ['datasets/kronodroid_emulador-balanced.csv'],
        "num_samples_class_benign": ['10000'],
        "num_samples_class_malware": ['10000'],
        'number_epochs': ['300'],
        'output_dir':["campanhas_demo"],
        'training_algorithm': ['Adam'],
    },
    'Kronodroid_r': {
        'input_dataset': ['datasets/kronodroid_emulador-balanced.csv'],
        "dense_layer_sizes_g": ['4096'],
        "dense_layer_sizes_d": ['2048'],
        "num_samples_class_benign": ['10000'],
        "num_samples_class_malware": ['10000'],
        'number_epochs': ['500'],
        'k_fold': ['10'],
        'output_dir':["campanhas_kronodroid_r"],
        'training_algorithm': ['Adam'],
    },
    'Kronodroid_e': {
        'input_dataset': ['datasets/kronodroid_real_device-balanced.csv'],
        "dense_layer_sizes_g": ['4096'],
        "dense_layer_sizes_d": ['2048'],
        'number_epochs': ['500'],
        'k_fold': ['10'],
        "num_samples_class_benign": ['10000'],
        "num_samples_class_malware": ['10000'],
        'output_dir':["campanhas_kronodroid_e"],
        'training_algorithm': ['Adam'],
    },
    'SF24_4096_2048_10': {
        'input_dataset': ['datasets/kronodroid_real_device-balanced.csv', 'datasets/kronodroid_emulador-balanced.csv'],
        "dense_layer_sizes_g": ['4096'],
        "dense_layer_sizes_d": ['2048'],
        'number_epochs': ['500'],
        'k_fold': ['10'],
        "num_samples_class_benign": ['10000'],
        "num_samples_class_malware": ['10000'],
        'output_dir':["campanhas_SF24"],
        'training_algorithm': ['Adam'],
    },
     'teste': {
        'input_dataset': ['datasets/kronodroid_real_device-balanced.csv', 'datasets/kronodroid_emulador-balanced.csv'],
        'number_epochs': ['100'],
        'k_fold': ['2'],
        "num_samples_class_benign": ['10000'],
        "num_samples_class_malware": ['10000'],
        'output_dir':["teste"],
        'training_algorithm': ['Adam'],
    },
}

def print_config(Parâmetros):
    """
    Imprime a configuração dos argumentos para fins de logging.

    Parâmetros:
        Parâmetros : Argumentos de linha de comando.
    """
    logging.info("Command:\n\t{0}\n".format(" ".join([x for x in sys.argv])))
    logging.info("Settings:")
    lengths = [len(x) for x in vars(Parâmetros).keys()]
    max_length = max(lengths)

    for k, v in sorted(vars(Parâmetros).items()):
        message = "\t" + k.ljust(max_length, " ") + " : {}".format(v)
        logging.info(message)
    logging.info("")

def convert_flot_to_int(value):
    """
    Converte um valor float para int multiplicando por 100.

    Parâmetros:
        value: Valor a ser convertido.

    Retorno:
        value: Valor convertido.
    """
    if isinstance(value, float):
        value = int(value * 100)
    return value

class IntRange:
    """
    Tipo personalizado de argparse que representa um inteiro delimitado por um intervalo.
    """
    def __init__(self, imin=None, imax=None):
        self.imin = imin
        self.imax = imax

    def __call__(self, arg):
        try:
            value = int(arg)
        except ValueError:
            raise self.exception()

        if (self.imin is not None and value < self.imin) or (self.imax is not None and value > self.imax):
            raise self.exception()

        return value

    def exception(self):
        if self.imin is not None and self.imax is not None:
            return argparse.ArgumentTypeError(f"Must be an integer in the range [{self.imin}, {self.imax}]")
        elif self.imin is not None:
            return argparse.ArgumentTypeError(f"Must be an integer >= {self.imin}")
        elif self.imax is not None:
            return argparse.ArgumentTypeError(f"Must be an integer <= {self.imax}")
        else:
            return argparse.ArgumentTypeError("Must be an integer")

def run_cmd(cmd, shell=False):
    """
    Executa um comando de shell e registra a saída.

    Parâmetros:
        cmd : Comando a ser executado.
        shell : Indica se deve usar o shell para executar o comando.
    """
    logging.info("Command line  : {}".format(cmd))
    cmd_array = shlex.split(cmd)
    logging.debug("Command array: {}".format(cmd_array))
    if not Parâmetros.demo:
        subprocess.run(cmd_array, check=True, shell=shell)

class Campaign:
    """
    Classe que representa uma campanha de treino.
    """
    def __init__(self, datasets, training_algorithm, dense_layer_sizes_g, dense_layer_sizes_d):
        self.datasets = datasets
        self.training_algorithm = training_algorithm
        self.dense_layer_sizes_g = dense_layer_sizes_g
        self.dense_layer_sizes_d = dense_layer_sizes_d

def check_files(files, error=False):
    """
    Verifica se os arquivos especificados existem.

    Parâmetros:
        files: Arquivos a verificar.
        error: Indica se deve lançar erro se o arquivo não for encontrado.

    Retorno:
        bool: True se todos os arquivos forem encontrados, False caso contrário.
    """
    internal_files = files if isinstance(files, list) else [files]

    for f in internal_files:
        if not os.path.isfile(f):
            if error:
                logging.info("ERROR: file not found! {}".format(f))
                sys.exit(1)
            else:
                logging.info("File not found! {}".format(f))
                return False
        else:
            logging.info("File found: {}".format(f))
    return True

def main():
    """
    Função principal que configura e executa as campanhas.
    """

    parser = argparse.ArgumentParser(description='Torrent Trace Correct - Machine Learning')
    #definição dos arugmentos de entrada
    help_msg = "Campaign {} (default={})".format([x for x in campaigns_available.keys()], DEFAULT_CAMPAIGN)
    parser.add_argument("--campaign", "-c", help=help_msg, default=DEFAULT_CAMPAIGN, type=str)
    parser.add_argument("--demo", "-d", help="demo mode (default=False)", action='store_true')
    help_msg = "verbosity logging level (INFO=%d DEBUG=%d)" % (logging.INFO, logging.DEBUG)
    parser.add_argument("--verbosity", "-v", help=help_msg, default=DEFAULT_VERBOSITY_LEVEL, type=int)
    parser.add_argument('-ml','--use_mlflow',action='store_true',help="Uso ou não da ferramenta mlflow para monitoramento") 

    global Parâmetros
    Parâmetros = parser.parse_args()
    #cria a estrutura dos diretórios de saída
    print("Creating the structure of directories...")
    for p in PATHS:
        Path(p).mkdir(parents=True, exist_ok=True)
    print("done.\n")
    output_dir = 'outputs/out_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logging_filename = '{}/evaluation_campaigns.log'.format(output_dir)

    logging_format = '%(asctime)s\t***\t%(message)s'
    if Parâmetros.verbosity == logging.DEBUG:
        logging_format = '%(asctime)s\t***\t%(levelname)s {%(module)s} [%(funcName)s] %(message)s'
    logging.basicConfig(format=logging_format, level=Parâmetros.verbosity)

    rotatingFileHandler = RotatingFileHandler(filename=logging_filename, maxBytes=100000, backupCount=5)
    rotatingFileHandler.setLevel(Parâmetros.verbosity)
    rotatingFileHandler.setFormatter(logging.Formatter(logging_format))
    logging.getLogger().addHandler(rotatingFileHandler)

    print_config(Parâmetros)
    # Tratamento das campanhas escolhidas
    campaigns_chosen = []
    if Parâmetros.campaign is None:
        campaigns_chosen = campaigns_available.keys()
    else:
        if Parâmetros.campaign in campaigns_available.keys():
            campaigns_chosen.append(Parâmetros.campaign)
        elif ',' in Parâmetros.campaign:
            campaigns_chosen = Parâmetros.campaign.split(',')
        else:
            logging.error("Campaign '{}' not found".format(Parâmetros.campaign))
            sys.exit(-1)
    # Obtém o tempo de início da execução
    time_start_campaign = datetime.datetime.now()
    logging.info("\n\n\n")
    logging.info("##########################################")
    logging.info(" EVALUATION ")
    logging.info("##########################################")
    time_start_evaluation = datetime.datetime.now()
    count_campaign = 1
    USE_MLFLOW=False
    #Estabelece o endereço de servidor de rastreamento mlflow como localhost porta 6002
    if Parâmetros.use_mlflow:
         USE_MLFLOW= True
    if USE_MLFLOW==False:
        for c in campaigns_chosen:
            #inicializa a execução mlflow

                logging.info("\tCampaign {} {}/{} ".format(c, count_campaign, len(campaigns_chosen)))
                count_campaign += 1
                campaign = campaigns_available[c]
                params, values = zip(*campaign.items())
                combinations_dicts = [dict(zip(params, v)) for v in itertools.product(*values)]
                campaign_dir = '{}/{}'.format(output_dir, c)
                count_combination = 1
                for combination in combinations_dicts:
                    logging.info("\t\tcombination {}/{} ".format(count_combination, len(combinations_dicts)))
                    logging.info("\t\t{}".format(combination))

                    cmd = COMMAND
                    cmd += " --verbosity {}".format(Parâmetros.verbosity)
                    cmd += " --output_dir {}".format(os.path.join(campaign_dir, "combination_{}".format(count_combination)))

                    count_combination += 2

                    for param in combination.keys():
                        cmd += " --{} {}".format(param, combination[param])

                    time_start_experiment = datetime.datetime.now()
                    logging.info("\t\t\t\t\tBegin: {}".format(time_start_experiment.strftime(TIME_FORMAT)))
                    run_cmd(cmd)

                    time_end_experiment = datetime.datetime.now()
                    duration = time_end_experiment - time_start_experiment
                    logging.info("\t\t\t\t\tEnd                : {}".format(time_end_experiment.strftime(TIME_FORMAT)))
                    logging.info("\t\t\t\t\tExperiment duration: {}".format(duration))

                time_end_campaign = datetime.datetime.now()
                logging.info("\t Campaign duration: {}".format(time_end_campaign - time_start_campaign))
        #Obtém o tempo de final da execução
        time_end_evaluation = datetime.datetime.now()
        logging.info("Evaluation duration: {}".format(time_end_evaluation - time_start_evaluation))
    else:
        count_campaign = 1
        mlflow.set_tracking_uri("http://127.0.0.1:5000/")
        mlflow.set_experiment("MalSynGEn")
        with mlflow.start_run(run_name="campanhas"): 
         for c in campaigns_chosen:
           with mlflow.start_run(run_name=c,nested=True) as run:
            id=run.info.run_id
            logging.info("\tCampaign {} {}/{} ".format(c, count_campaign, len(campaigns_chosen)))
            count_campaign += 1

            campaign = campaigns_available[c]
            params, values = zip(*campaign.items())
            combinations_dicts = [dict(zip(params, v)) for v in itertools.product(*values)]
            campaign_dir = '{}/{}'.format(output_dir, c)

            count_combination = 1
            for combination in combinations_dicts:
                logging.info("\t\tcombination {}/{} ".format(count_combination, len(combinations_dicts)))
                logging.info("\t\t{}".format(combination))

                cmd = COMMAND2
                cmd += " --verbosity {}".format(Parâmetros.verbosity)
                cmd += " --output_dir {}".format(os.path.join(campaign_dir, "combination_{}".format(count_combination)))
                #cmd +=" -ml"
                cmd += " --run_id {}".format(id)
                #cmd +=" -ml"
                count_combination += 2

                for param in combination.keys():
                    cmd += " --{} {}".format(param, combination[param])


                time_start_experiment = datetime.datetime.now()
                logging.info(
                    "\t\t\t\t\tBegin: {}".format(time_start_experiment.strftime(TIME_FORMAT)))
                run_cmd(cmd)

                time_end_experiment = datetime.datetime.now()
                duration = time_end_experiment - time_start_experiment
                logging.info("\t\t\t\t\tEnd                : {}".format(time_end_experiment.strftime(TIME_FORMAT)))
                logging.info("\t\t\t\t\tExperiment duration: {}".format(duration))


            time_end_campaign = datetime.datetime.now()
            logging.info("\t Campaign duration: {}".format(time_end_campaign - time_start_campaign))

        time_end_evaluation = datetime.datetime.now()
        logging.info("Evaluation duration: {}".format(time_end_evaluation - time_start_evaluation))



if __name__ == '__main__':
    sys.exit(main())
