__author__ = "Giovanni Da San Martino"
__copyright__ = "Copyright 2019"
__credits__ = [""]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = ""
__email__ = "gmartino@hbku.edu.qa"
__status__ = "Beta"

import sys
import argparse
import os.path
import re
import glob
import logging.handlers

TASK_3_ARTICLE_ID_COL=0
#TASK_3_TECHNIQUE_NAME_COL=1
TASK_3_FRAGMENT_START_COL=1
TASK_3_FRAGMENT_END_COL=2
TECHNIQUE_NAMES_FILE="propaganda-techniques-names.txt"

logger = logging.getLogger("propaganda_scorer")
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.setLevel(logging.INFO)


def check_data_file_lists(submission_annotations, gold_annotations, task_name="task3"):

    #checking that the number of articles for which the user has submitted annotations is correct
    if len(gold_annotations.keys()) < len(submission_annotations.keys()):
        logger.error("The number of articles in the submission, %d, is greater than the number of articles in the "
                     "reference dataset, %d." % (len(submission_annotations.keys()), len(gold_annotations.keys()))); sys.exit()
    # logger.debug("OK: number of articles in the submission for %s is the same as the one in the gold file: %d"
    #              % (task_name, len(gold_annotations.keys())))

    #check that all file names are correct
    errors = [ article_id for article_id in submission_annotations.keys() if article_id not in gold_annotations.keys() ]
    if len(errors)>0:
        logger.error("The following article_ids in the submission file do not have a correspondence in the reference "
                     "dataset: %s\n" % (str(errors)))
    # gold_file_article_id_set = set([article_id for article_id in gold_annotations.keys()])
    # submission_file_article_id_set = set([article_id for article_id in submission_annotations.keys()])
    # intersection_file_list = gold_file_article_id_set.intersection(submission_file_article_id_set)
    # if len(intersection_file_list) != len(gold_annotations):
    #     logger.error("The list of article ids is not identical.\n"
    #              "The following article_ids in the submission file do not have a correspondence in the gold file: %s\n"
    #              "The following article_ids in the gold file do not have a correspondence in the submission file: %s"
    #              %(str(submission_file_article_id_set.difference(gold_file_article_id_set)),
    #                str(gold_file_article_id_set.difference(submission_file_article_id_set)))); sys.exit()
    logger.debug("OK: all article ids have a correspondence in the list of articles from the reference dataset")


def load_technique_names_from_file(filename=TECHNIQUE_NAMES_FILE):

    with open(filename, "r") as f:
        return [ line.rstrip() for line in f.readlines() ]


def extract_article_id_from_file_name(fullpathfilename):

    regex = re.compile("article([0-9]+).*")
    return regex.match(os.path.basename(fullpathfilename)).group(1)


def print_annotations(annotation_list):
    s = ""
    i=0
    for technique, span in annotation_list:
        s += "%d) %s: %d - %d\n"%(i, technique, min(span), max(span))
        i += 1
    return s

def merge_spans(annotations_without_overlapping, i):
    """

    :param spans: a list of spans of article annotations
    :param i: the index in spans which needs to be tested for ovelapping
    :param annotations: a list of annotations of an article
    :return:
    """
    #print("checking element %d of %d"%(i, len(spans)))
    for j in range(0, len(annotations_without_overlapping)):
        assert i<len(annotations_without_overlapping) or print(i, len(annotations_without_overlapping))
        if j != i and len(annotations_without_overlapping[i][1].intersection(annotations_without_overlapping[j][1])) > 0:
            # print("Found overlapping spans: %d-%d and %d-%d in annotations %d,%d:\n%s"
            #       %(min(annotations_without_overlapping[i][1]), max(annotations_without_overlapping[i][1]),
            #         min(annotations_without_overlapping[j][1]), max(annotations_without_overlapping[j][1]), i,j,
            #         print_annotations(annotations_without_overlapping)))
            annotations_without_overlapping[j][1] = annotations_without_overlapping[j][1].union(annotations_without_overlapping[i][1])
            del(annotations_without_overlapping[i])
            # print("Annotations after deletion:\n%s"%(print_annotations(annotations_without_overlapping)))
            if j > i:
                j -= 1
            # print("calling recursively")
            merge_spans(annotations_without_overlapping, j)
            # print("done")
            return True

    return False


def check_annotation_spans(annotations, merge_overlapping_spans=False):

    for article_id in annotations.keys():  # for each article
        #print(article_id)
        spans = []
        annotations_without_overlapping = []
        for annotation in annotations[article_id]:
            if merge_overlapping_spans:
                #spans.append(annotation[1])
                annotations_without_overlapping.append([annotation[0], annotation[1]])
                #if merge_spans(spans, len(spans)-1, annotations[article_id]):
                merge_spans(annotations_without_overlapping, len(annotations_without_overlapping) - 1)
                #if merge_spans(annotations_without_overlapping, len(annotations_without_overlapping) - 1):
                #    print("Done with merging:\n" + print_annotations(annotations_without_overlapping))
            else:
                for span in spans:
                    if len(span.intersection(annotation[1])) > 0:
                        logger.error("In article %s, the span [%s,%s] overlap with the following one from the same "
                                     "article: [%s,%s]" % (article_id, min(annotation[1]), max(annotation[1]),
                                                           min(span), max(span)))
                        return False
                spans.append(annotation[1])
        if merge_overlapping_spans:
            annotations[article_id] = annotations_without_overlapping
    return True


def check_annotation_spans_with_category_matching(annotations, merge_overlapping_spans=False):
    """
    Check whether there are ovelapping spans for the same technique in the same article.
    Two spans are overlapping if their associated techniques match (according to category_matching_func)
    If merge_overlapping_spans==True then the overlapping spans are merged, otherwise an error is raised.

    :param annotations: a dictionary with the full set of annotations for all articles
    :param merge_overlapping_spans: if True merges the overlapping spans
    :return:
    """

    for article_id in annotations.keys():  # for each article

        annotation_list = {}
        for technique, curr_span in annotations[article_id]:
            if technique not in annotation_list.keys():
                annotation_list[technique] = [ [technique, curr_span] ]
            else:
                if merge_overlapping_spans:
                    annotation_list[technique].append([technique, curr_span])
                    merge_spans(annotation_list[technique], len(annotation_list[technique]) - 1)
                else:
                    for matching_technique, span in annotation_list[technique]:
                        if len(curr_span.intersection(span)) > 0:
                            logger.error("In article %s, the span of the annotation %s, [%s,%s] overlap with "
                                         "the following one from the same article:%s, [%s,%s]" % (article_id, matching_technique,
                                                                min(span), max(span), technique, min(curr_span), max(curr_span)))
                            return False
                    annotation_list[technique].append([technique, curr_span])
        if merge_overlapping_spans:
            annotations[article_id] = []
            for technique in annotation_list.keys():
                annotations[article_id] += annotation_list[technique]
    return True


def check_format_of_annotation_in_file(row, i, techniques_names, filename):
    """

    :param row: a list of fields describing the annotation elements (technique name, start_span, end_span)
    :param i:
    :return:
    """

    if len(row) != 3:
        logger.error("Row %d in file %s is supposed to have 3 TAB-separated columns. Found %d."
                     % (i + 1, filename, len(row)))
        sys.exit()
    # checking the technique names are correct
    # if row[TASK_3_TECHNIQUE_NAME_COL] not in techniques_names:
    #     logger.error("On row %d in file %s the technique name, %s, is incorrect. Possible values are: %s"
    #                  % (i + 1, filename, row[TASK_3_TECHNIQUE_NAME_COL], str(techniques_names)))
    #     sys.exit()
    # checking spans
    if int(row[TASK_3_FRAGMENT_START_COL]) < 0 or int(row[TASK_3_FRAGMENT_END_COL]) < 0:
        logger.error("On row %d in file %s, start and end of position of the fragment must be non-negative. "
                     "Found values %s, %s" % (i + 1, filename, row[TASK_3_FRAGMENT_START_COL], row[TASK_3_FRAGMENT_END_COL]))
        sys.exit()
    if int(row[TASK_3_FRAGMENT_START_COL]) >= int(row[TASK_3_FRAGMENT_END_COL]):
        logger.error("On row %d in file %s, end position of the fragment must be greater than the starting "
                     "one. Found values %s, %s" % (i + 1, filename, row[TASK_3_FRAGMENT_START_COL], row[TASK_3_FRAGMENT_END_COL]))
        sys.exit()


def check_article_annotations_format(submission_article_annotations, article_id, techniques_names):

    annotations = {}
    for i, row in enumerate(submission_article_annotations):
        check_annotation_format_from_file(row, i)
        #checking that there are no overlapping spans flagged with the same technique name
        if row[0] not in annotations.keys():
            annotations[row[0]] = []
        else:
            curr_span = set(range(int(row[1]), int(row[2])))
            for span in annotations[row[0]]:
                if len(set(range(int(span[0]), int(span[1]))).intersection(curr_span)) > 0:
                    logger.error("On row %d in article %s, the span of the annotation %s, [%s,%s] overlap with the "
                                 "following one from the same file: [%s,%s]"
                                 % (i + 1, article_id, row[0], row[1], row[2], span[0], span[1]));
                    sys.exit()
        annotations[row[0]].append([row[1], row[2]])
    logger.debug("OK: article %s format is correct" % (article_id))


def read_task3_output_file(filename):

    with open(filename, "r") as f:
        return [ line.rstrip().split("\t") for line in f.readlines() ]


def compute_technique_frequency(annotations_list, technique_name):
    return sum([ len([ example_annotation for example_annotation in x if example_annotation[0]==technique_name])
                 for x in annotations_list ])


def compute_score_max(submission_annotations, gold_annotations, technique_names, prop_vs_non_propaganda=False):

    prec_denominator = sum([len(annotations) for annotations in submission_annotations.values()])
    rec_denominator = sum([len(annotations) for annotations in gold_annotations.values()])
    technique_Spr_max = {propaganda_technique: 0 for propaganda_technique in technique_names}
    cumulative_Spr_max = 0
    for article_id in submission_annotations.keys():
        gold_data = gold_annotations[article_id]
        logger.debug("Computing contribution to the score of article id %s\nand tuples %s\n%s\n"
                     % (article_id, str(submission_annotations[article_id]), str(gold_data)))
        for j, sd in enumerate(submission_annotations[article_id]): #submission_data:
            s=""
            sd_annotation_length = len(sd[1])
            for i, gd in enumerate(gold_data):
                if prop_vs_non_propaganda or gd[0]==sd[0]:
                    #s += "\tmatch %s %s-%s - %s %s-%s"%(sd[0],sd[1], sd[2], gd[0], gd[1], gd[2])
                    intersection = len(sd[1].intersection(gd[1]))
                    gd_annotation_length = len(gd[1])

                    Spr = intersection/max(sd_annotation_length, gd_annotation_length)
                    cumulative_Spr_max += Spr
                    s += "\tmatch %s %s-%s - %s %s-%s: S(p,r)=|intersect(r, p)|/max(|p|,|r|) = %d/max(%d,%d) = %f (cumulative S(p,r)=%f)\n"\
                         %(sd[0],min(sd[1]), max(sd[1]), gd[0], min(gd[1]), max(gd[1]), intersection, sd_annotation_length, gd_annotation_length, Spr, cumulative_Spr_max)
                    technique_Spr_max[gd[0]] += Spr

            logger.debug("\n%s"%(s))

    p,r,f1 = compute_prec_rec_f1(cumulative_Spr_max, prec_denominator, cumulative_Spr_max, rec_denominator)

    if not prop_vs_non_propaganda:
        for technique_name in cumulative_Spr_max.keys():
            prec_tech, rec_tech, f1_tech = compute_prec_rec_f1(cumulative_Spr_max[technique_name],
                                        compute_technique_frequency(submission_annotations.values(), technique_name),
                                        cumulative_Spr_max[technique_name],
                                        compute_technique_frequency(gold_annotations.values(), technique_name), False)
            logger.info("%s: P=%f R=%f F1=%f" % (technique_name, prec_tech, rec_tech, f1_tech))

    # p,r,f1=(0,0,0)
    # if prec_denominator>0:
    #     p = cumulative_Spr_max/prec_denominator
    # if rec_denominator>0:
    #     r = cumulative_Spr_max/rec_denominator
    # logger.info("Precision=%f/%d=%f\tRecall=%f/%d=%f"
    #              %(cumulative_Spr_max, prec_denominator, p, cumulative_Spr_max, rec_denominator, r))
    # if prec_denominator == 0 and rec_denominator == 0:
    #     f1 = 1.0
    # if p>0 and r>0:
    #     f1 = 2*(p*r/(p+r))
    # logger.info("F1=%f"%(f1))
    #
    # if not prop_vs_non_propaganda:
    #     for technique_name in technique_Spr_max.keys():
    #         prec_tech, rec_tech, f1_tech = (0,0,0)
    #         prec_tech_denominator = compute_technique_frequency(submission_annotations.values(), technique_name)
    #         rec_tech_denominator = compute_technique_frequency(gold_annotations.values(), technique_name)
    #         if prec_tech_denominator == 0 and rec_tech_denominator == 0: #
    #             f1_tech = 1.0
    #         else:
    #             if prec_tech_denominator > 0:
    #                 prec_tech = technique_Spr_max[technique_name] / prec_tech_denominator
    #             if rec_tech_denominator > 0:
    #                 rec_tech = technique_Spr_max[technique_name] / rec_tech_denominator
    #             if prec_tech>0 and rec_tech>0:
    #                 f1_tech = 2*(prec_tech*rec_tech/(prec_tech+rec_tech))
    #         logger.info("%s: P=%f R=%f F1=%f"%(technique_name, prec_tech, rec_tech, f1_tech))

    return f1

def compute_score_min(submission_annotations, gold_annotations, technique_names, prop_vs_non_propaganda=False):

    prec_denominator = sum([len(annotations) for annotations in submission_annotations.values()])
    rec_denominator = sum([len(annotations) for annotations in gold_annotations.values()])
    technique_Spr_min = {propaganda_technique: 0 for propaganda_technique in technique_names}
    cumulative_Spr_min = 0
    for article_id in submission_annotations.keys():
        gold_data = gold_annotations[article_id]
        logger.debug("Computing contribution to the score of article id %s\nand tuples %s\n%s\n"
                     % (article_id, str(submission_annotations[article_id]), str(gold_data)))
        for j, sd in enumerate(submission_annotations[article_id]): #submission_data:
            s=""
            sd_annotation_length = len(sd[1])
            for i, gd in enumerate(gold_data):
                if prop_vs_non_propaganda or gd[0]==sd[0]:
                    #s += "\tmatch %s %s-%s - %s %s-%s"%(sd[0],sd[1], sd[2], gd[0], gd[1], gd[2])
                    intersection = len(sd[1].intersection(gd[1]))
                    gd_annotation_length = len(gd[1])

                    Spr = intersection/min(sd_annotation_length, gd_annotation_length)
                    cumulative_Spr_min += Spr
                    s += "\tmatch %s %s-%s - %s %s-%s: S(p,r)=|intersect(r, p)|/min(|p|,|r|) = %d/min(%d,%d) = %f (cumulative S(p,r)=%f)\n"\
                         %(sd[0],min(sd[1]), max(sd[1]), gd[0], min(gd[1]), max(gd[1]), intersection, sd_annotation_length, gd_annotation_length, Spr, cumulative_Spr_min)
                    technique_Spr_min[gd[0]] += Spr

            logger.debug("\n%s"%(s))

    p,r,f1 = compute_prec_rec_f1(cumulative_Spr_min, prec_denominator, cumulative_Spr_min, rec_denominator)

    if not prop_vs_non_propaganda:
        for technique_name in cumulative_Spr_min.keys():
            prec_tech, rec_tech, f1_tech = compute_prec_rec_f1(cumulative_Spr_min[technique_name],
                                        compute_technique_frequency(submission_annotations.values(), technique_name),
                                        cumulative_Spr_min[technique_name],
                                        compute_technique_frequency(gold_annotations.values(), technique_name), False)
            logger.info("%s: P=%f R=%f F1=%f" % (technique_name, prec_tech, rec_tech, f1_tech))

    # p,r,f1=(0,0,0)
    # if prec_denominator>0:
    #     p = cumulative_Spr_min/prec_denominator
    # if rec_denominator>0:
    #     r = cumulative_Spr_min/rec_denominator
    # logger.info("Precision=%f/%d=%f\tRecall=%f/%d=%f"
    #              %(cumulative_Spr_min, prec_denominator, p, cumulative_Spr_min, rec_denominator, r))
    # if prec_denominator == 0 and rec_denominator == 0:
    #     f1 = 1.0
    # if p>0 and r>0:
    #     f1 = 2*(p*r/(p+r))
    # logger.info("F1=%f"%(f1))
    #
    # if not prop_vs_non_propaganda:
    #     for technique_name in technique_Spr_min.keys():
    #         prec_tech, rec_tech, f1_tech = (0,0,0)
    #         prec_tech_denominator = compute_technique_frequency(submission_annotations.values(), technique_name)
    #         rec_tech_denominator = compute_technique_frequency(gold_annotations.values(), technique_name)
    #         if prec_tech_denominator == 0 and rec_tech_denominator == 0: #
    #             f1_tech = 1.0
    #         else:
    #             if prec_tech_denominator > 0:
    #                 prec_tech = technique_Spr_min[technique_name] / prec_tech_denominator
    #             if rec_tech_denominator > 0:
    #                 rec_tech = technique_Spr_min[technique_name] / rec_tech_denominator
    #             if prec_tech>0 and rec_tech>0:
    #                 f1_tech = 2*(prec_tech*rec_tech/(prec_tech+rec_tech))
    #         logger.info("%s: P=%f R=%f F1=%f"%(technique_name, prec_tech, rec_tech, f1_tech))

    return f1


def compute_score_pr(submission_annotations, gold_annotations, technique_names, prop_vs_non_propaganda=False,
                     per_article_evaluation=False, output_for_script=False):

    prec_denominator = sum([len(annotations) for annotations in submission_annotations.values()])
    rec_denominator = sum([len(annotations) for annotations in gold_annotations.values()])
    technique_Spr_prec = {propaganda_technique: 0 for propaganda_technique in technique_names}
    technique_Spr_rec = {propaganda_technique: 0 for propaganda_technique in technique_names}
    cumulative_Spr_prec, cumulative_Spr_rec = (0, 0)
    f1_articles = []

    for article_id in submission_annotations.keys():
        gold_data = gold_annotations.get(article_id, [])
        logger.debug("Computing contribution to the score of article id %s\nand tuples %s\n%s\n"
                     % (article_id, str(submission_annotations[article_id]), str(gold_data)))

        article_cumulative_Spr_prec, article_cumulative_Spr_rec = (0, 0)
        for j, sd in enumerate(submission_annotations[article_id]): #submission annotations for article article_id:
            s=""
            sd_annotation_length = len(sd[1])
            for i, gd in enumerate(gold_data):
                if prop_vs_non_propaganda or gd[0]==sd[0]:
                    #s += "\tmatch %s %s-%s - %s %s-%s"%(sd[0],sd[1], sd[2], gd[0], gd[1], gd[2])
                    intersection = len(sd[1].intersection(gd[1]))
                    gd_annotation_length = len(gd[1])
                    Spr_prec = intersection/sd_annotation_length
                    article_cumulative_Spr_prec += Spr_prec
                    cumulative_Spr_prec += Spr_prec
                    s += "\tmatch %s %s-%s - %s %s-%s: S(p,r)=|intersect(r, p)|/|p| = %d/%d = %f (cumulative S(p,r)=%f)\n"\
                         %(sd[0],min(sd[1]), max(sd[1]), gd[0], min(gd[1]), max(gd[1]), intersection, sd_annotation_length, Spr_prec, cumulative_Spr_prec)
                    technique_Spr_prec[gd[0]] += Spr_prec

                    Spr_rec = intersection/gd_annotation_length
                    article_cumulative_Spr_rec += Spr_rec
                    cumulative_Spr_rec += Spr_rec
                    s += "\tmatch %s %s-%s - %s %s-%s: S(p,r)=|intersect(r, p)|/|r| = %d/%d = %f (cumulative S(p,r)=%f)\n"\
                         %(sd[0],min(sd[1]), max(sd[1]), gd[0], min(gd[1]), max(gd[1]), intersection, gd_annotation_length, Spr_rec, cumulative_Spr_rec)
                    technique_Spr_rec[gd[0]] += Spr_rec
            logger.debug("\n%s"%(s))

        p_article, r_article, f1_article =compute_prec_rec_f1(article_cumulative_Spr_prec,
                                                              len(submission_annotations[article_id]),
                                                              article_cumulative_Spr_rec,
                                                              len(gold_annotations.get(article_id, [])), False)
        f1_articles.append(f1_article)

    p,r,f1 = compute_prec_rec_f1(cumulative_Spr_prec, prec_denominator, cumulative_Spr_rec, rec_denominator)
    out = ""
    if output_for_script:
        out += "%f\t%f\t%f"%(f1,p,r) 
    if not prop_vs_non_propaganda:
        for technique_name in technique_names:
            prec_tech, rec_tech, f1_tech = compute_prec_rec_f1(technique_Spr_prec[technique_name],
                                        compute_technique_frequency(submission_annotations.values(), technique_name),
                                        technique_Spr_prec[technique_name],
                                        compute_technique_frequency(gold_annotations.values(), technique_name), False)
            logger.info("%s: P=%f R=%f F1=%f" % (technique_name, prec_tech, rec_tech, f1_tech))
            if output_for_script:
                f1_str = str(f1_tech)
                if len(f1_str) > 1:
                    out += "\t" + f1_str.rstrip("0").rstrip(".")
                else:
                    out += "\t" + f1_str
    if output_for_script:
        print(out)
    if per_article_evaluation:
        logger.info("Per article evaluation F1=%s"%(",".join([ str(f1_value) for f1_value in  f1_articles])))

    return f1


def compute_prec_rec_f1(prec_numerator, prec_denominator, rec_numerator, rec_denominator, print_results=True):

    logger.debug("P=%f/%d, R=%f/%d"%(prec_numerator, prec_denominator, rec_numerator, rec_denominator))
    p, r, f1 = (0, 0, 0)
    if prec_denominator > 0:
        p = prec_numerator / prec_denominator
    if rec_denominator > 0:
        r = rec_numerator / rec_denominator
    if print_results: logger.info("Precision=%f/%d=%f\tRecall=%f/%d=%f" % (prec_numerator, prec_denominator, p,
                                                                           rec_numerator, rec_denominator, r))
    if prec_denominator == 0 and rec_denominator == 0:
        f1 = 1.0
    if p > 0 and r > 0:
        f1 = 2 * (p * r / (p + r))
    if print_results:
        logger.info("F1=%f" % (f1))
    return p,r,f1


def load_annotation_list_from_folder(folder_name, techniques_names):

    file_list = glob.glob(os.path.join(folder_name, "*.labels"))
    if len(file_list)==0:
        logger.error("Cannot load file list in folder " + folder_name);
        sys.exit()
    annotations = {}
    for filename in file_list:
        annotations[extract_article_id_from_file_name(filename)] = []
        with open(filename, "r") as f:
            for row_number, line in enumerate(f.readlines()):
                row = line.rstrip().split("\t")
                check_format_of_annotation_in_file(row, row_number, techniques_names, filename)
                # annotations[row[TASK_3_ARTICLE_ID_COL]].append([ row[TASK_3_TECHNIQUE_NAME_COL],
                #                                                  row[TASK_3_FRAGMENT_START_COL],
                #                                                  row[TASK_3_FRAGMENT_END_COL] ])
                annotations[row[TASK_3_ARTICLE_ID_COL]].append(["propaganda",
                                                                set(range(int(row[TASK_3_FRAGMENT_START_COL]),
                                                                          int(row[TASK_3_FRAGMENT_END_COL])))])
    return annotations


def load_annotation_list_from_file(filename, techniques_names):

    annotations = {}
    with open(filename, "r") as f:
        for row_number, line in enumerate(f.readlines()):
            row = line.rstrip().split("\t")
            check_format_of_annotation_in_file(row, row_number, techniques_names, filename)
            if row[TASK_3_ARTICLE_ID_COL] not in annotations.keys():
                annotations[row[TASK_3_ARTICLE_ID_COL]] = []
            # annotations[row[TASK_3_ARTICLE_ID_COL]].append([ row[TASK_3_TECHNIQUE_NAME_COL],
            #                                                  row[TASK_3_FRAGMENT_START_COL],
            #                                                  row[TASK_3_FRAGMENT_END_COL]])
            annotations[row[TASK_3_ARTICLE_ID_COL]].append([ "propaganda",
                                                             set(range(int(row[TASK_3_FRAGMENT_START_COL]),
                                                                       int(row[TASK_3_FRAGMENT_END_COL]))) ])
    return annotations


def main(args):

    user_submission_file = args.submission
    gold_folder = args.gold
    output_log_file = args.log_file
    prop_vs_non_propaganda = True
    merge_user_annotations = bool(args.merge_user_annotations)
    per_article_evaluation = False
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

    techniques_names = [ "propaganda" ] #load_technique_names_from_file(args.techniques_file)
    submission_annotations = load_annotation_list_from_file(user_submission_file, techniques_names)
    logger.info('Checking user submitted file')
    gold_annotations = load_annotation_list_from_file(gold_folder, techniques_names)
    check_data_file_lists(submission_annotations, gold_annotations)
    if not check_annotation_spans(submission_annotations, merge_user_annotations):
        logger.info("Error in submission file")
        sys.exit()
    check_annotation_spans(gold_annotations, True)

    logger.info("Scoring the submission with precision and recall method")
    score_pr = compute_score_pr(submission_annotations, gold_annotations, techniques_names,
                                prop_vs_non_propaganda, per_article_evaluation, output_for_script)
    return score_pr


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Scorer for Task SI on the Propaganda Techniques Corpus. ")
    parser.add_argument('-s', '--submission-file', dest='submission', required=True, help="file with the submission of the team")
    parser.add_argument('-r', '--reference-folder', dest='gold', required=True, help="folder with the gold labels.")
    parser.add_argument('-d', '--enable-debug-on-standard-output', dest='debug_on_std', required=False,
                        action='store_true', help="Print debug info also on standard output.")
    parser.add_argument('-l', '--log-file', dest='log_file', required=False, help="Output logger file.")
    parser.add_argument('-m', '--merge-user-annotations', dest='merge_user_annotations', required=False, action='store_true',
                        default=False, help="If the option is added, overlapping user annotations are merged")
    parser.add_argument('-o', '--output-for-script', dest='output_for_script', required=False, action='store_true',
                        default=False, help="Prints the output in an easy-to-parse way for a script")
    main(parser.parse_args())
