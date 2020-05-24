try:
    from .transformers_classifier import transformers_clf
    from .dataset import load_data, get_train_dev_files, get_test_file
    from .submission import create_submission_file, eval_submission
except:
    from transformers_classifier import transformers_clf
    from dataset import load_data, get_train_dev_files, get_test_file
    from submission import create_submission_file, eval_submission
    
import configargparse
import logging
import os
import subprocess


logger = logging.getLogger(__name__)


def Main(args):
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    
    if args.do_train or args.do_eval or args.split_dataset or args.create_submission_file:
        articles, ref_articles_id, ref_span_starts, ref_span_ends, labels = load_data(args.train_data_folder, 
                                                                           args.labels_path)
        train_file_path = os.path.join(args.data_dir, args.train_file)
        dev_file_path = os.path.join(args.data_dir, args.dev_file)
        if not os.path.exists(train_file_path) or not os.path.exists(dev_file_path) or args.overwrite_cache:
            logger.info("Creating train/dev files: %s, %s", train_file_path, dev_file_path)
            get_train_dev_files(articles, ref_articles_id, ref_span_starts, ref_span_ends, labels, train_file_path, 
                                dev_file_path, args.split_by_ids, args.dev_size, args.random_state, args.balance,
                                args.shuffle)
    
    if args.do_predict or args.create_submission_file or args.eval_submission:
        test_file_path = os.path.join(args.data_dir, args.test_file)
        test_articles, test_articles_id, test_span_starts, test_span_ends, test_labels = load_data(args.test_data_folder,
                                                                                      args.test_template_labels_path)
        if not os.path.exists(test_file_path) or args.overwrite_cache:
            logger.info("Creating roberta-type test file: %s", test_file_path)
            get_test_file(test_articles, test_articles_id, test_span_starts, test_span_ends, test_labels, test_file_path)
           
    if args.do_train or args.do_eval or args.do_predict:
        transformers_clf(args)
    
    if args.create_submission_file:
        if not os.path.exists('results'):
            os.makedirs('results')
        output_file = os.path.join('results', args.output_file)
        logger.info("Creating the submission file: %s", output_file)        
        create_submission_file(args.predicted_logits_files, train_file_path, dev_file_path, test_file_path, 
                            test_articles_id, test_span_starts, test_span_ends, output_file, args.weights, args.data_dir)
        
    if args.eval_submission:
        output_file = os.path.join('results', args.output_file)
        logger.info("Evaluating the submission file: %s", output_file)
        if args.test_labels_path is None:
            acc, f1 = eval_submission(output_file, test_file_path)
            logger.info('accuracy: %f', acc)
            print('f1-macro:', f1)
        else:
            cmd = "python tools/task-TC_scorer.py -s {} -r {} -p {}".format(output_file, args.test_labels_path,
                                                                          args.propaganda_techniques_file)
            subprocess.run(cmd, shell=True)


def main(): 
    parser = configargparse.ArgumentParser()
    
    parser.add_argument('--config', required=True, is_config_file=True, help='Config file path.')
    parser.add_argument("--train_data_folder", default=None, type=str, required=True,
                        help="Source directory with the train articles.")
    parser.add_argument("--test_data_folder", default=None, type=str, required=True,
                        help="Source directory with the test articles.")
    parser.add_argument("--propaganda_techniques_file", default=None, type=str, required=True,
                    help="The file with propaganda techniques.")
    parser.add_argument("--labels_path", default=None, type=str, required=True,
                        help="The file with train labels.")
    parser.add_argument("--test_template_labels_path", default=None, type=str, required=True,
                        help="The file with test template labels.")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The directory for cached preprocessed data.")
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="The filename for cached preprocessed train data.")
    parser.add_argument("--dev_file", default=None, type=str, required=True,
                        help="The filename for cached preprocessed dev data.")
    parser.add_argument("--test_file", default=None, type=str, required=True,
                        help="The filename for cached preprocessed test data.")
    parser.add_argument("--predicted_logits_files", default=None, nargs='*', required=True,
                        help="The predicted filenames of logits that will be used to obtain the final result")
    parser.add_argument("--weights", default=None, nargs='*', required=False,
                        help="The list of weights for predicted logits at the aggregation stage")
    parser.add_argument("--output_file", default=None, type=str, required=True,
                        help="The submission filename")
    parser.add_argument("--dev_size", default=0.3, type=float, help="Dev data size.")
    parser.add_argument("--split_dataset", action="store_true", 
                        help="Split the dataset into the train/dev parts.")
    parser.add_argument("--split_by_ids", action="store_true", 
                        help="Use articles ids while splitting the dataset into the train/dev parts.")
    parser.add_argument("--random_state", default=42, type=int, help='Random state for the dataset splitting.')
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the train dataset.")
    parser.add_argument("--balance", action="store_true", help="Balance the train dataset with oversampling.")
    parser.add_argument("--create_submission_file", action="store_true", 
                        help="Creats file in the submission (source) format")
    parser.add_argument("--eval_submission", action="store_true", help="Do evaluating for the dev subset.")
    
    parser.add_argument('--use_length', action='store_true')
    parser.add_argument('--join_embeddings', action='store_true')
    parser.add_argument('--use_matchings', action='store_true')
    
    MODEL_CLASSES = ["bert", "roberta", "distilbert", "camembert"]
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name.")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--test_labels_path", default=None, type=str, required=False)

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action='store_true',
                        help="Whether to run prediction")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")     
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()
    
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    
    Main(args)
    
    
if __name__ == "__main__":
    main()
