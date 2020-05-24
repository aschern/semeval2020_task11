try:
    from .ner import transformers_ner_crf, transformers_ner
    from .dataset import load_data, get_train_dev_files, get_test_file, create_subfolder
    from .submission import get_submission_format
except:
    from ner import transformers_ner_crf, transformers_ner
    from dataset import load_data, get_train_dev_files, get_test_file, create_subfolder
    from submission import get_submission_format
    
import configargparse
import spacy
import logging
import os
import subprocess
import tempfile


logger = logging.getLogger(__name__)


def Main(args):
    nlp = spacy.load("en")
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    
    if args.do_train or args.do_eval or args.split_dataset:
        articles_content, articles_id, propaganda_techniques_names = load_data(args.train_data_folder, 
                                                                           args.propaganda_techniques_file)
        train_file_path = os.path.join(args.data_dir, args.train_file)
        dev_file_path = os.path.join(args.data_dir, args.dev_file)
        if not os.path.exists(train_file_path) or not os.path.exists(dev_file_path) or args.overwrite_cache:
            logger.info("Creating 'ner' train/dev files: %s, %s", train_file_path, dev_file_path)
            train_ids, dev_ids = get_train_dev_files(articles_id, articles_content, nlp, args.labels_path, train_file_path,
                                                     dev_file_path, args.split_by_ids, args.dev_size, args.random_state)
            if args.split_dataset:
                create_subfolder(os.path.join(args.data_dir, 'train-train-articles'),  args.train_data_folder, train_ids)
                create_subfolder(os.path.join(args.data_dir, 'train-dev-articles'),  args.train_data_folder, dev_ids)
    
    if args.do_predict or args.create_submission_file or args.do_eval_spans:
        test_articles_content, test_articles_id, _ = load_data(args.test_data_folder, args.propaganda_techniques_file)
        test_file_path = os.path.join(args.data_dir, args.test_file)
        if (not os.path.exists(test_file_path) or args.overwrite_cache) and not args.do_eval_spans:
            logger.info("Creating 'ner' test file: %s", test_file_path)
            get_test_file(test_file_path, test_articles_id, test_articles_content, nlp)            
    
    if args.do_train or args.do_eval or args.do_predict:
        if args.use_crf:
            transformers_ner_crf(args)
        else:
            transformers_ner(args)
            
    if args.do_eval_spans:
        logger.info("Evaluating file %s with competition metrics", args.output_file)
        output_file = os.path.join('results', args.output_file)
        get_submission_format(args.predicted_labels_files, test_articles_id, test_articles_content, nlp, output_file)
        if args.gold_annot_file is None:
            gold_annot_file = next(tempfile._get_candidate_names())
            get_submission_format([test_file_path], test_articles_id, test_articles_content, nlp, gold_annot_file)
        else:
            gold_annot_file = args.gold_annot_file
        cmd = "python tools/task-SI_scorer.py -s {} -r {}".format(output_file, gold_annot_file)
        subprocess.run(cmd, shell=True)
        if args.gold_annot_file is None:
            os.remove(gold_annot_file)
    
    if args.create_submission_file:
        if not os.path.exists('results'):
            os.makedirs('results')
        output_file = os.path.join('results', args.output_file)
        logger.info("Creating a submission file: %s", output_file)
        get_submission_format(args.predicted_labels_files, test_articles_id, test_articles_content, nlp, output_file)


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
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The directory for cached preprocessed data.")
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="The filename for cached preprocessed train data.")
    parser.add_argument("--dev_file", default=None, type=str, required=True,
                        help="The filename for cached preprocessed dev data.")
    parser.add_argument("--test_file", default=None, type=str, required=True,
                        help="The filename for cached preprocessed test data.")
    parser.add_argument("--predicted_labels_files", default=None, nargs='*', required=True,
                        help="The predicted filenames of labels that will be used to form the final result")
    parser.add_argument("--output_file", default=None, type=str, required=True,
                        help="The submission filename")
    parser.add_argument("--dev_size", default=0.3, type=float, help="Dev data size.")
    parser.add_argument("--split_dataset", action="store_true", 
                        help="Split the dataset into the train/dev parts")
    parser.add_argument("--split_by_ids", action="store_true", 
                        help="Use articles ids while splitting the dataset into the train/dev parts.")
    parser.add_argument("--create_submission_file", action="store_true", 
                        help="Creats file in the submission (source) format")
    parser.add_argument("--random_state", default=42, type=int, help='Random state for the dataset splitting.')
    parser.add_argument("--do_eval_spans", action="store_true", 
                        help="Whether to run eval on the dev set with the competition metrics.")
    parser.add_argument("--gold_annot_file", default=None, type=str, help="Gold annotation file.")

    parser.add_argument("--use_crf", action="store_true", help="Use Conditional Random Field over the model")
    parser.add_argument("--use_quotes", action="store_true")
    
    MODEL_CLASSES = ["bert", "roberta", "distilbert", "camembert"]
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--labels", default="", type=str,
                        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true",
                        help="Whether to run predictions on the test set.")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
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

    parser.add_argument("--logging_steps", type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action="store_true",
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()
    
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    
    Main(args)
    
    
if __name__ == "__main__":
    main()
