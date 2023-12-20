from configs.base import ParamManager, add_config_param
from data.base import DataManager
from methods import method_map
from utils.functions import set_torch_seed, save_results, set_output_path

import argparse
import logging
import os
import datetime
import itertools
import warnings
import copy

def parse_arguments():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--logger_name', type=str, default='Multimodal Intent Recognition', help="Logger name for multimodal intent recognition.")

    parser.add_argument('--dataset', type=str, default='MIntRec', help="The selected person id.")

    parser.add_argument('--data_mode', type=str, default='multi-class', help="The selected person id.")

    parser.add_argument('--method', type=str, default='TCL-MAP', help="which method to use.")

    parser.add_argument("--text_backbone", type=str, default='bert-base-uncased', help="which backbone to use for text modality")

    parser.add_argument('--seed', type=int, default=0, help="The selected person id.")

    parser.add_argument('--num_workers', type=int, default=8, help="The number of workers to load data.")

    parser.add_argument('--log_id', type=str, default=None, help="The index of each logging file.")
    
    parser.add_argument('--gpu_id', type=str, default='0', help="The selected person id.")

    parser.add_argument("--data_path", default = '/Datasets', type=str,
                        help="The input data dir. Should contain text, video and audio data for the task.")

    parser.add_argument("--train", action="store_true", help="Whether to train the model.")

    parser.add_argument("--tune", action="store_true", help="Whether to tune the model with a series of hyper-parameters.")

    parser.add_argument("--save_model", action="store_true", help="save trained-model for multimodal intent recognition.")

    parser.add_argument("--save_results", action="store_true", help="save final results for multimodal intent recognition.")

    parser.add_argument('--log_path', type=str, default='logs', help="Logger directory.")
    
    parser.add_argument('--cache_path', type=str, default='cache', help="The caching directory for pre-trained models.")   

    parser.add_argument('--results_path', type=str, default='results', help="The path to save results.")

    parser.add_argument("--output_path", default= 'outputs', type=str, 
                        help="The output directory where all train data will be written.") 

    parser.add_argument("--model_path", default= 'models', type=str, 
                        help="The output directory where the model predictions and checkpoints will be written.") 

    parser.add_argument("--config_file_name", type=str, default='TCL_MAP_MIntRec.py', help = "The name of the config file.")

    parser.add_argument("--results_file_name", type=str, default = 'results.csv', help="The file name of all the results.")    
    
    parser.add_argument('--save_pred', type=bool, default=False, help="Logger directory.")
    

    args = parser.parse_args()

    return args

def set_logger(args):
    
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    
    time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    args.logger_name =  f"{args.method}_{args.dataset}_{args.data_mode}_{args.seed}"
    args.log_id = f"{args.logger_name}_{time}"
    
    logger = logging.getLogger(args.logger_name)
    logger.setLevel(logging.DEBUG)

    log_path = os.path.join(args.log_path, args.log_id + '.log')
    fh = logging.FileHandler(log_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(fh_formatter)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)
    
    return logger

def set_up(args):
    

    save_model_name = f"{args.method}_{args.dataset}_{args.text_backbone}_{args.data_mode}_{args.seed}"
    
    args.pred_output_path, args.model_output_path = set_output_path(args, save_model_name)
    
    set_torch_seed(args.seed)
    
    return args
    
def work(args, data, logger, debug_args=None, ind_args = None):
    
    set_torch_seed(args.seed)
    
    method_manager = method_map[args.method]
    method = method_manager(args, data)
        
    
    logger.info('Intent Recognition begins...')

    if args.train:

        logger.info('Training begins...')
        method._train(args)
        logger.info('Training is finished...')

    logger.info('Testing begins...')
    outputs = method._test(args)
    logger.info('Testing is finished...')
    logger.info('Intent recognition is finished...')
    
    if args.save_results:
        
        logger.info('Results are saved in %s', str(os.path.join(args.results_path, args.results_file_name)))
        save_results(args, outputs, debug_args=debug_args)
        
def run(args, data, logger, ind_args = None):
    debug_args = {}

    for k,v in args.items():
        if isinstance(v, list):
            debug_args[k] = v
        
    for result in itertools.product(*debug_args.values()):
        for i, key in enumerate(debug_args.keys()):
            args[key] = result[i]         
        
        work(args, data, logger, debug_args, ind_args)

if __name__ == '__main__':
    
    warnings.filterwarnings('ignore')
    
    args = parse_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    param = ParamManager(args)
    args = param.args
    args = add_config_param(args, args.config_file_name)
    args = set_up(args)
    
    data = DataManager(args)
    logger = set_logger(args)
    
    
    logger = set_logger(args)
    logger.info("="*30+" Params "+"="*30)
    for k in args.keys():
        logger.info(f"{k}: {args[k]}")
    logger.info("="*30+" End Params "+"="*30)
    
    run(args, data, logger)
    

