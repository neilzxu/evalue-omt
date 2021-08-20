import argparse
import multiprocess as mp
from exp.utils import get_experiment
# yapf: disable
if __name__ == '__main__':
    mp.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument('--processes',
                        type=int,
                        default=1,
                        help="size of process pool used to run experiments.")
    parser.add_argument('--exp',
                        required=True,
                        type=str,
                        help="name of experiment to run")
    parser.add_argument('--out_dir',
                        required=True,
                        type=str,
                        help="Directory to refer to/write results to")
    parser.add_argument('--result_dir',
                        required=True,
                        type=str,
                        help="Directory to save plots to")
    # flag for parser
    parser.add_argument('--no_save_result',
            default=True,
            action='store_false',
                        help="Don't save individual results of each experiment")
    parser.add_argument('--custom_param',
            default=None,
            type=str,
                        help="Custom parameter to break down experiments that are too large")
    args = parser.parse_args()
    get_experiment(args.exp)(processes=args.processes, out_dir=args.out_dir, result_dir=args.result_dir, save_result=args.no_save_result, custom_param=args.custom_param)
