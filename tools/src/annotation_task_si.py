import sys
import src.annotation as an

class AnnotationTaskSI(Annotation)

    def __init__(self, label=None, start_offset = None, end_offset=None): #, article_id=None):
        
        self.label = label
        self.start_offset = int(start_offset)
        self.end_offset = int(end_offset)


    def get_label(self):

        sys.error("ERRRO: trying to access technique label from file in SI task format")


