from __future__ import annotations
import sys
import logging.handlers
import src.propaganda_techniques as pt
import src.annotation_w_o_label as anwol

__author__ = "Giovanni Da San Martino"
__copyright__ = "Copyright 2019"
__credits__ = ["Giovanni Da San Martino"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Giovanni Da San Martino"
__email__ = "gmartino@hbku.edu.qa"
__status__ = "Beta"

logger = logging.getLogger("propaganda_scorer")


class Annotation(anwol.AnnotationWithOutLabel):

    """
    One annotation is represented by a span (two integer indices indicating the 
    starting and ending position of the span) and the propaganda technique name 
    (a label attached to the span). 
    The class provides basic maniputation functions for one annotation. 
    """

    # input file format variables
    separator = "\t"
    ARTICLE_ID_COL = 0
    TECHNIQUE_NAME_COL = 1
    FRAGMENT_START_COL = 2
    FRAGMENT_END_COL = 3
    propaganda_techniques:pt.Propaganda_Techniques = None


    def __init__(self, label:str=None, start_offset:str = None, end_offset:str=None): 
        
        super().__init__(start_offset, end_offset)
        self.label = label


    def __str__(self):

        return super().__str__() + " -> " + self.get_label()
        #return self.get_label() + "\t" + super().__str__()


    def __eq__(self, second_annotation:Annotation):
        """
        Checks whether two annotations are identical, i.e. if their spans are 
        identical and if they labels coincide
        """        
        return super().__eq__(second_annotation) and self.get_label()==second_annotation.get_label()


    def get_label(self)->str:

        return self.label

    
    def get_propaganda_techniques(self)->list:

        if self.propaganda_techniques is None:
            logger.error("trying to access propaganda techniques list before initialising the corresponding object")
            sys.exit()
        return self.propaganda_techniques.get_propaganda_techniques_list()

    
    @classmethod
    def set_propaganda_technique_list_obj(cls, propaganda_technique_obj:pt.Propaganda_Techniques)->None:
        """
        propaganda_technique_obj is an object from the module src.propaganda_techniques.
        Typical invokation: 
        `
            propaganda_techniques = pt.Propaganda_Techniques(filename=propaganda_techniques_list_file)
            an.Annotation.set_propaganda_technique_list_obj(propaganda_techniques)
        `
        """
        cls.propaganda_techniques = propaganda_technique_obj


    @staticmethod
    def load_annotation_from_string(annotation_string:str, row_num:int=None, filename:str=None)->(Annotation, str):
        """
        Read annotations from a csv-like string, with fields separated
        by the class variable `separator`: 

        article id<separator>technique name<separator>starting_position<separator>ending_position
        Fields order is determined by the class variables ARTICLE_ID_COL,
        TECHNIQUE_NAME_COL, FRAGMENT_START_COL, FRAGMENT_END_COL

        Besides reading the data, it performs basic checks.

        :return a tuple (Annotation object, id of the article)
        """

        row = annotation_string.rstrip().split(Annotation.separator)
        if len(row) != 4:
            logger.error("Row%s%s is supposed to have 4 columns. Found %d: -%s-."
                         % (" " + str(row_num) if row_num is not None else "",
                            " in file " + filename if filename is not None else "", len(row), annotation_string))
            sys.exit()

        article_id = row[Annotation.ARTICLE_ID_COL]
        label = row[Annotation.TECHNIQUE_NAME_COL]
        try:
            start_offset = int(row[Annotation.FRAGMENT_START_COL])
        except:
            logger.error("The column %d in row%s%s is supposed to be an integer: -%s-"
                         %(Annotation.FRAGMENT_START_COL, " " + str(row_num) if row_num is not None else "",
                            " in file " + filename if filename is not None else "", annotation_string))
        try:
            end_offset = int(row[Annotation.FRAGMENT_END_COL])
        except:
            logger.error("The column %d in row%s%s is supposed to be an integer: -%s-"
                         %(Annotation.FRAGMENT_END_COL, " " + str(row_num) if row_num is not None else "",
                            " in file " + filename if filename is not None else "", annotation_string))

        return Annotation(label, start_offset, end_offset), article_id
        

    def is_technique_name_valid(self)->bool:
        """
        Checks whether the technique names are correct
        """
        if self.propaganda_techniques is None:
            sys.exit("ERROR: propaganda techniques object has not been initialised")
        if not self.propaganda_techniques.is_valid_technique(self.get_label()):
            logger.error("label %s is not valid. Possible values are: %s"%(self.get_label(), self.propaganda_techniques))
            return False
        return True


    def check_format_of_annotation_in_file(self):
        """
        Performs some checks on the fields of the annotation
        """
        if not self.is_technique_name_valid():
            sys.exit()
        if not self.is_span_valid():
            sys.exit()

