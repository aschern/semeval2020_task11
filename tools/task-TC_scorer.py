import sys
import argparse
import logging.handlers
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import src.annotation as an
import src.annotations as ans
import src.propaganda_techniques as pt

logger = logging.getLogger("propaganda_scorer")
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.setLevel(logging.INFO)


def main(args):

    user_submission_file = args.submission
    gold_file = args.gold
    output_log_file = args.log_file
    propaganda_techniques_list_file = args.propaganda_techniques_list_file
    output_for_script = bool(args.output_for_script)

    if not output_for_script:
        logger.addHandler(ch)

    if args.debug_on_std:
        ch.setLevel(logging.DEBUG)

    if output_log_file is not None:
        logger.info("Logging execution to file " + output_log_file)
        fileLogger = logging.FileHandler(output_log_file)
        fileLogger.setLevel(logging.DEBUG)
        fileLogger.setFormatter(formatter)
        logger.addHandler(fileLogger)

    propaganda_techniques = pt.Propaganda_Techniques(propaganda_techniques_list_file)
    an.Annotation.set_propaganda_technique_list_obj(propaganda_techniques)

    user_annotations = ans.Annotations()
    user_annotations.load_annotation_list_from_file(user_submission_file)
    for article in user_annotations.get_article_id_list():
        user_annotations.get_article_annotations_obj(article).sort_spans()

    gold_annotations = ans.Annotations()
    gold_annotations.load_annotation_list_from_file(gold_file)
    for article in gold_annotations.get_article_id_list():
        gold_annotations.get_article_annotations_obj(article).sort_spans()

    logger.info("Checking format: User Predictions -- Gold Annotations")
    if not user_annotations.compare_annotations_identical_article_lists(gold_annotations) or not user_annotations.compare_annotations_identical(gold_annotations):
        logger.error("wrong format, no scoring will be performed")
        sys.exit()
    logger.info("OK: submission file format appears to be correct")
    res_for_output, res_for_script = user_annotations.TC_score_to_string(gold_annotations, output_for_script)
    logger.info("Scoring submission" + res_for_output)
    if output_for_script:
        print(res_for_script)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Scorer for SemEval 2020 Task 11 subtask TC.\n" +
    "Example: python3 task-TC_scorer.py -s data/submission-task-TC.tsv -r data/article736757214.task-FLC.labels -p data/propaganda-techniques-names-semeval2020task11.txt")

    parser.add_argument('-s', '--submission-file', dest='submission', required=True, help="file with the submission of the team")
    parser.add_argument('-r', '--reference-file', dest='gold', required=True, help="file with the gold labels.")
    parser.add_argument('-d', '--enable-debug-on-standard-output', dest='debug_on_std', required=False,
                        action='store_true', help="Print debug info also on standard output.")
    parser.add_argument('-l', '--log-file', dest='log_file', required=False, help="Output logger file.")
    parser.add_argument('-p', '--propaganda-techniques-list-file', dest='propaganda_techniques_list_file', required=True, 
                        help="file with list of propaganda techniques (one per line).")
    parser.add_argument('-o', '--output-for-script', dest='output_for_script', required=False, action='store_true',
                        default=False, help="Prints the output in a format easy to parse for a script")
    main(parser.parse_args())
