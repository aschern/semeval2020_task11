from __future__ import annotations
import sys
import src.propaganda_techniques as pt
import logging.handlers

__author__ = "Giovanni Da San Martino"
__copyright__ = "Copyright 2019"
__credits__ = ["Giovanni Da San Martino"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Giovanni Da San Martino"
__email__ = "gmartino@hbku.edu.qa"
__status__ = "Beta"

logger = logging.getLogger("propaganda_scorer")


class AnnotationWithOutLabel(object):

    """
    One annotation is represented by a span (two integer indices indicating the 
    starting and ending position of the span). 
    The class provides basic maniputation functions for one annotation. 
    """

    # input file format variables
    separator = "\t"
    ARTICLE_ID_COL = 0
    FRAGMENT_START_COL = 1
    FRAGMENT_END_COL = 2


    def __init__(self, start_offset:str = None, end_offset:str=None): 
        
        self.start_offset = int(start_offset)
        self.end_offset = int(end_offset)


    def __str__(self):

        return "[%d, %d]"%(self.start_offset, self.end_offset)
        #return "%d\t%d"%(self.start_offset, self.end_offset)


    def is_span_equal_to(self, second_annotation:AnnotationWithOutLabel)->bool:
        """
        Checks whether two annotations are identical, i.e. whether the two spans are identical. 
        """
        if self.get_start_offset() != second_annotation.get_start_offset() or self.get_end_offset() != second_annotation.get_end_offset():
            return False
        return True


    def __eq__(self, second_annotation:AnnotationWithOutLabel):
        
        return self.is_span_equal_to(second_annotation)


    def get_start_offset(self)->int:

        return self.start_offset

        
    def get_end_offset(self)->int:

        return self.end_offset

    
    def get_span(self)->set:
        """
        Returns a set of positions of all characters in the span
        """
        return set(range(self.get_start_offset(), self.get_end_offset()))

    
    @staticmethod
    def load_annotation_from_string(annotation_string:str, row_num:int=None, filename:str=None)->(AnnotationWithOutLabel, str):
        """
        Read annotations from a csv-like string, with fields separated
        by the class variable `separator`: 

        article id<separator>starting_position<separator>ending_position
        Fields order is determined by the class variables ARTICLE_ID_COL,
        FRAGMENT_START_COL, FRAGMENT_END_COL

        Besides reading the data, it performs basic checks.

        :return a tuple (AnnotationWithOutLabel object, id of the article)
        """

        row = annotation_string.rstrip().split(AnnotationWithOutLabel.separator)
        if len(row) != 3:
            logger.error("Row%s%s is supposed to have 3 columns. Found %d: -%s-."
                         % (" " + str(row_num) if row_num is not None else "",
                            " in file " + filename if filename is not None else "", len(row), annotation_string))
            sys.exit()

        article_id = row[AnnotationWithOutLabel.ARTICLE_ID_COL]
        try:
            start_offset = int(row[AnnotationWithOutLabel.FRAGMENT_START_COL])
        except:
            logger.error("The column %d in row%s%s is supposed to be an integer: -%s-"
                         %(AnnotationWithOutLabel.FRAGMENT_START_COL, " " + str(row_num) if row_num is not None else "", " in file " + filename if filename is not None else "", annotation_string))
        try:
            end_offset = int(row[AnnotationWithOutLabel.FRAGMENT_END_COL])
        except:
            logger.error("The column %d in row%s%s is supposed to be an integer: -%s-"
                         %(AnnotationWithOutLabel.FRAGMENT_END_COL, " " + str(row_num) if row_num is not None else "",
                            " in file " + filename if filename is not None else "", annotation_string))

        return AnnotationWithOutLabel(start_offset, end_offset), article_id


    def merge_spans(self, second_annotation:AnnotationWithOutLabel)->None:
        """
        Merge the spans of two annotations. The function does not check whether the spans overlap. 

        :param second_annotation: the AnnotationWithOutLabel object whose span is being merged
        :return:
        """
        self.set_start_offset(min(self.get_start_offset(), second_annotation.get_start_offset()))
        self.set_end_offset(max(self.get_end_offset(), second_annotation.get_end_offset()))


    def set_start_offset(self, new_start_offset:int)->None:

        self.start_offset = new_start_offset


    def set_end_offset(self, new_end_offset:int)->None:

        self.end_offset = new_end_offset


    def shift_annotation(self, offset:int)->None:
        
        self.set_start_offset(self.get_start_offset() + offset)
        self.set_end_offset(self.get_end_offset() + offset)
        

    def span_overlapping(self, second_annotation:AnnotationWithOutLabel)->bool:
        return len(self.get_span().intersection(second_annotation.get_span())) > 0


    def is_span_valid(self)->bool:
        """
        Checks whether the span is valid, i.e. if the following conditions are met: 
        1) start and end offsets >= 0 
        2) start offset < end offset
        """
        if self.get_start_offset() < 0 or self.get_end_offset() < 0:
            logger.error("Start and end of position of the fragment must be non-negative: %d, %d"
                         %(self.get_start_offset(), self.get_end_offset()))
            return False
        if self.get_start_offset() >= self.get_end_offset():
            logger.error("End position of the fragment must be greater than the starting one: start=%d, end=%d"%(self.get_start_offset(), self.get_end_offset()))
            return False
        return True
        

    def check_format_of_annotation_in_file(self):
        """
        Performs some checks on the fields of the annotation
        """
        if not self.is_span_valid():
            sys.exit()

