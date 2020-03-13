from __future__ import annotations
import sys
import logging.handlers
import re
import os.path
import src.annotation as ans
import src.annotation_w_o_label as anwol
from src.propaganda_techniques import Propaganda_Techniques

__author__ = "Giovanni Da San Martino"
__copyright__ = "Copyright 2019"
__credits__ = ["Giovanni Da San Martino"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Giovanni Da San Martino"
__email__ = "gmartino@hbku.edu.qa"
__status__ = "Beta"

logger = logging.getLogger("propaganda_scorer")


class Articles_annotations(object):

    """
    Class for handling annotations for one article. 
    Articles_annotations is composed of an article id
    and a list of Annotation (or AnnotationWithOutLabel) objects. 
    """

    start_annotation_effect = "\033[42;33m"
    end_annotation_effect = "\033[m"
    start_annotation_str = "{"
    end_annotation_str = "}"
    annotation_background_color = "\033[44;33m"
    inadmissible_annotation_boundary_chars = [ " ", "\n", ".", ",", ":", ";" ]
    techniques:Propaganda_Techniques = None #Propaganda_Techniques()


    def __init__(self, spans:anwol.AnnotationWithOutLabel=None, article_id:str=None):

        if spans is None:
            self.spans = []
        else:
            self.spans = spans
        self.article_id = article_id


    def __len__(self):

        return len(self.spans)


    def __str__(self):

        return "article id: %s\n%s"%(self.article_id, "\n".join([ "\t" + str(annotation) for annotation in self.spans ]))


    def add_annotation(self, annotation, article_id:str=None):
        """
        param annotation: an Annotation object
        """
        if article_id is None:
            article_id = self.get_article_id()
        self.add_article_id(article_id)
        #if not isinstance(annotation, Annotation):
        #    sys.exit()
        self.spans.append(annotation)


    def add_article_id(self, article_id:str):

        if self.article_id is None:
            self.article_id = article_id
        else:
            if article_id is not None and self.article_id != article_id:
                logger.error("Trying to add an annotation with a different article id")
                sys.exit()


    def _matching_annotations(self, second_annotations:Articles_annotations, i:int)->list(int, int):

            j2 = i
            while j2<len(second_annotations) and second_annotations[j2].is_span_equal_to(second_annotations[i]):
                j1 = i
                while j1<len(self) and self[j1].is_span_equal_to(self[i]):
                    if self[j1]==second_annotations[j2]:
                        return j1, j2
                    j1 += 1
                j2 += 1
            return -1, -1


    def adapt_annotation_to_new_text(self, original_text:str, new_text:str)->Articles_annotations:
        """
        create a copy of the annotations such that they refer to new_text (a slightly modified version of original_text, the text the current annotation refers to). 
        This function currently assumes that original_text has only additional spaces w.r.t new_text. 
        """
        i,j=(0,0)
        #a_txt = a_txt.replace('\'\'', '"')

        while i<len(original_text) and j<len(new_text):
            o, n = original_text[i], new_text[j]
            if n==o or (o in ['"','“','”'] and n in ['"', '“', '”']) \
                or (o in ['\'','’', '`'] and n in ['\'','’', '`']):
                i+=1
                j+=1
            else:
                if original_text[i] != " ":
                    print("different", original_text[i], original_text[i].encode(encoding='UTF-8'), new_text[j])
                    sys.exit()
                i+=1                    
                self.shift_spans(j,-1)                


    def align_annotations(self, second_annotations:Articles_annotations)->None:
        """
        Reorder all annotations such that the matching between annotations' labels
        and the ones from second_annotations is maximised. 
        Annotations are modified in place
        """
        i, j1, j2 = 0, 0, 0
        while i < len(self):
            #if not self[j].is_span_equal_to(second_annotations[i]): 
            #    logger.error("trying to align annotations with different spans: %s, %s"%(self[j], second_annotations[i]))
            #    sys.exit()
            j1, j2 = self._matching_annotations(second_annotations, i)
            if j1>i:
                self.swap_annotations(j1, i)
            if j2>i:
                second_annotations.swap_annotations(j2, i)
            i+=1
            if j1==-1: # no match, can forward i skipping all annotations with same span
                while i+1<len(self) and self[i].is_span_equal_to(self[i+1]):
                    i+=1


    def _get_annotation_offset_excluding_chars(self, article_content:str, fragment_index:int, is_starting_fragment:bool):

        new_offset = 0
        try:
            while article_content[fragment_index + new_offset + (0 if is_starting_fragment else -1)] in self.inadmissible_annotation_boundary_chars:
                if is_starting_fragment:
                    new_offset += 1
                else: 
                    new_offset -= 1
        except:
            print("accessing char %d+%d+%d=%d of string of len %d"%(fragment_index, new_offset, (0 if is_starting_fragment else -1), fragment_index + new_offset + (0 if is_starting_fragment else -1), len(article_content)))
            sys.exit()
        return new_offset


    def check_article_annotations_out_of_boundaries(self, article_content:str, fix_errors:bool=False)->list(bool, bool):
        """
        Check that all annotations for the article do not reference fragments beyond article length
        """
        annotations_correct = True
        need_saving = False
        annotations_to_be_removed = []
        for ann in self.get_article_annotations():
            if len(article_content) < ann.get_end_offset(): # making sure the offsets of the annotation are within the length of the article content
                logger.error("trying to access fragment beyond article length: %s, article %s length: %d"%(ann, self.get_article_id(), len(article_content)))
                if fix_errors:
                    if ann.get_start_offset() > len(article_content)-1: # the annotation is completely beyond the length of the article
                        annotations_to_be_removed.append(ann)
                        #self.remove_annotation(ann)
                    else:
                        ann.set_end_offset(len(article_content))
                    need_saving = True
                else:
                    annotations_correct = False

        for ann in annotations_to_be_removed:
            self.remove_annotation(ann)

        return annotations_correct, need_saving


    def check_article_annotations_start_end_chars(self, article_content:str, fix_errors:bool=False)->list(bool, bool):
        """
        Check that all annotations for the article do not start or end with any of the characters
        specified in the class variable `inadmissible_annotation_boundary_chars`. 
        If fix_errors==True, tries to correct the spans of the wrong annotations.
        return: annotations_correct (True if the annotations did not have any issue or the issues have been fixed), annotations_need_saving (True if the annotation has been modified and need to be saved)
        """
        annotations_need_saving = False
        annotations_correct = True
        for ann in self.get_article_annotations():
            # check that the annotation does not start or end with an inadmissible char
            new_start_offset = self._get_annotation_offset_excluding_chars(article_content, ann.get_start_offset(), True)
            new_end_offset = self._get_annotation_offset_excluding_chars(article_content, ann.get_end_offset(), False)
            if new_start_offset != 0 or new_end_offset != 0:
                logger.error('annotation %s in article %s starts or ends with inadmissible characters: -%s-'%(ann, self.get_article_id(), article_content[ann.get_start_offset():ann.get_end_offset()]))
                annotations_correct = False
                if fix_errors:
                    ann.set_start_offset(ann.get_start_offset() + new_start_offset)
                    ann.set_end_offset(ann.get_end_offset() + new_end_offset)
                    if not ann.is_span_valid():
                        logger.error("impossible to fix annotation")
                        sys.exit()
                    else:
                        annotations_correct = True
                        annotations_need_saving = True
                
        return annotations_correct, annotations_need_saving


    def check_article_annotations_no_duplicated_annotations(self,article_content:str, fix_errors:bool=False)->bool:
        """
        Check that there are no two duplicated annotations. If there are such annotations and if they have identical labels, one is removed and the annotation is considered correct
        """
        annotations_correct = True
        annotations_need_saving = False
        for i, ann in enumerate(self.get_article_annotations()):
            # check that the annotation is not identical to any other annotation of the same article
            j = i + 1
            while j < len(self.get_article_annotations()):
                if ann.is_span_equal_to(self[j]):
                    logger.error("article %s found two identical annotations: %s and %s"%(self.get_article_id(), ann, self[j]))
                    if ann.get_label()==self[j].get_label() and fix_errors:
                        self.remove_annotation(self[j])
                        logger.debug("The two annotations have identical labels (%s, %s), deleted one."%(ann, self[j]))
                        annotations_need_saving = True
                j += 1
        return annotations_correct, annotations_need_saving


    def check_article_annotations(self, article_content:str, fix_errors:bool=False, 
                                  check_out_of_boundaries=False, check_start_end_chars=False, check_duplicated_annotations=False)->bool:
        """
        Performs the following checks on the annotations of the article:
        1) check that an annotation does not reference a location in article_content beyond its length
        2) check that an annotation does not start or end with a space, a full stop or a newline character. 
        3) there are no two annotations with identical spans 

        return value: annotations_correct (True if the annotations did not have any issue or the issues have been fixed), 
                      annotations_need_saving (True if the annotation has been modified and need to be saved)
        """

        annotations_correct, annotations_need_saving = True, False

        if check_out_of_boundaries:
            annotations_correct, annotations_need_saving = self.check_article_annotations_out_of_boundaries(article_content, fix_errors)

        if check_start_end_chars:
            start_end_chars_correct, need_saving = self.check_article_annotations_start_end_chars(article_content, fix_errors)
            annotations_correct = annotations_correct and start_end_chars_correct
            annotations_need_saving = annotations_need_saving or need_saving

        if check_duplicated_annotations:
            no_dup_annotations, dup_annotations_need_saving = self.check_article_annotations_no_duplicated_annotations(article_content, fix_errors)
            annotations_need_saving = annotations_need_saving or dup_annotations_need_saving
            annotations_correct = annotations_correct and no_dup_annotations

        return annotations_correct, annotations_need_saving


    def get_article_id(self)->str:

        return self.article_id


    def get_article_annotations(self)->list(anwol.AnnotationWithOutLabel):
        """
        returns a list of AnnotationWithoutLabel objects
        """
        return self.spans

        
    def __getitem__(self, index):
        return self.spans[index]


    def get_markers_from_spans(self):

        self.sort_spans()
        self.markers = []
        for i, annotation in enumerate(self.spans, 1):
            self.markers.append((annotation.get_start_offset(), annotation.get_label(), i, "start"))
            self.markers.append((annotation.get_end_offset(), annotation.get_label(), i, "end"))
        self.markers = sorted(self.markers, key=lambda ann: ann[0])


    def groupby_technique(self):

        annotation_list = {}
        for i, annotation in enumerate(self.get_article_annotations()):
            technique = annotation.get_label()
            if technique not in annotation_list.keys():
                annotation_list[technique] = []
            annotation_list[technique].insert(0, i)
        return annotation_list


    #check_annotation_spans_with_category_matching
    def has_overlapping_spans(self, prop_vs_non_propaganda, merge_overlapping_spans=False):
        """
        Check whether there are ovelapping spans for the same technique in the same article.
        Two spans are overlapping if their associated techniques match (according to category_matching_func)
        If merge_overlapping_spans==True then the overlapping spans are merged, otherwise an error is raised.

        :param merge_overlapping_spans: if True merges the overlapping spans
        :return:
        """

        annotation_list = {}
        for annotation in self.get_article_annotations():
            if prop_vs_non_propaganda:
                technique = "propaganda"
            else:
                technique = annotation.get_label()
            if technique not in annotation_list.keys():
                annotation_list[technique] = [annotation] #[[technique, curr_span]]
            else:
                if merge_overlapping_spans:
                    annotation_list[technique].append(annotation)
                    self.merge_article_annotations(annotation_list[technique], len(annotation_list[technique]) - 1)
                else:
                    for matching_annotation in annotation_list[technique]:
                        if annotation.span_overlapping(matching_annotation):
                            logger.error("In article %s, the span of the annotation %s, [%s,%s] overlap with "
                                         "the following one from the same article:%s, [%s,%s]" % (
                                             self.get_article_id(), annotation.get_label(),
                                             annotation.get_start_offset(), annotation.get_end_offset(), matching_annotation.get_label(), matching_annotation.get_start_offset(), matching_annotation.get_end_offset()))
                            return False
                    annotation_list[technique].append([annotation])
        if merge_overlapping_spans: # recreate the list of annotations
            self.reset_annotations()
            for anlist in annotation_list.values():
                for a in anlist: 
                    self.add_annotation(a)
        return True


    def is_starting_marker(self, marker_index=None):

        if marker_index is None:
            marker_index = self.curr_marker
        if marker_index < len(self.markers):
            return self.markers[marker_index][3] == "start"


    def is_ending_marker(self, marker_index=None):

        if marker_index is None:
            marker_index = self.curr_marker
        if marker_index < len(self.markers):
            return self.markers[marker_index][3] == "end"


    def load_article_annotations_from_csv_file(self, filename:str, annotation_class=ans.Annotation):
        """
        Read annotations from a csv file and add the annotations 
        found in the file as a list of Annotation objects. 
        Check class Annotation for details on the format of the file.
        """
        with open(filename, "r") as f:
            for i, line in enumerate(f.readlines(), 1):
                #an, article_id = ans.Annotation.load_annotation_from_string(line.rstrip(), i, filename)
                an, article_id = annotation_class.load_annotation_from_string(line.rstrip(), i, filename)
                self.add_annotation(an, article_id)


    def marker_label(self, marker_index=None):

        if marker_index is None:
            marker_index = self.curr_marker
        if marker_index < len(self.markers):
            return self.markers[marker_index][1]
        # else:
        # ERROR


    def marker_position(self, marker_index=None):

        if marker_index is None:
            marker_index = self.curr_marker
        if marker_index < len(self.markers):
            return self.markers[marker_index][0]


    def marker_annotation(self, marker_index=None):

        if marker_index is None:
            marker_index = self.curr_marker
        if marker_index < len(self.markers):
            return self.markers[marker_index][2]


    def mark_text(self, original_text, print_line_numbers=False):
        """
        mark the string original_text with object's annotations

        original_text: string with the text to be marked
        print_line_numbers: add line numbers to the text

        :return output_text the text in string original_text with added marks
                footnotes the list of techniques in the text
                legend description of the marks added
        """

        self.get_markers_from_spans()
        if Articles_annotations.techniques is None:
            if ans.Annotation.propaganda_techniques is None:
                Articles_annotations.techniques = Propaganda_Techniques()
            else:   
                Articles_annotations.techniques = ans.Annotation.propaganda_techniques

        output_text, curr_output_text_index, self.curr_marker = ("", 0, 0)
        footnotes = "List of techniques found in the article\n\n"
        techniques_found = set()
        annotations_stack = []  # to handle overlapping annotations when assigning color background
        while curr_output_text_index < len(original_text):
            if self.curr_marker >= len(self.markers):
                output_text += original_text[curr_output_text_index:]
                curr_output_text_index = len(original_text)
            else:
                if self.marker_position() <= curr_output_text_index:
                    if self.is_starting_marker():
                        output_text += self.start_annotation_effect + self.start_annotation_str
                        annotations_stack.append(self.marker_annotation())
                    else:
                        output_text += "%s%s%s" % (
                            self.end_annotation_effect, "" if len(annotations_stack) > 1 else " ",
                            self.start_annotation_effect)
                    techniques_index = Articles_annotations.techniques.indexOf(self.marker_label())
                    output_text += str(techniques_index)
                    techniques_found.add(techniques_index)
                    if self.is_ending_marker():
                        output_text += self.end_annotation_str + self.end_annotation_effect
                        annotations_stack.remove(self.marker_annotation())
                        if len(annotations_stack) > 0:
                            output_text += self.annotation_background_color
                    else:
                        output_text += self.end_annotation_effect + " " + self.annotation_background_color
                    self.curr_marker += 1
                else:
                    output_text += original_text[curr_output_text_index:self.marker_position()]
                    curr_output_text_index = self.marker_position()

        if print_line_numbers:
            indices, char_index = ([], 0)
            for line in original_text.split("\n"):
                indices.append(char_index)
                char_index += len(line) + 1
            #output_text = "\n".join(["%d (%d) %s"%(i, x[0], x[1])
            output_text = "\n".join(["%d %s"%(i, x[1]) 
                                     for i, x in enumerate(zip(indices, output_text.split("\n")), 1)])

        legend = "---\n%sHighlighted text%s: any propagandistic fragment\n%s%si%s: start of the i-th technique" \
                 "\n%si%s%s: end of the i-th technque\n---"\
                 %(self.annotation_background_color, self.end_annotation_effect, self.start_annotation_effect,
                   self.start_annotation_str, self.end_annotation_effect, self.start_annotation_effect,
                   self.end_annotation_str, self.end_annotation_effect)

        for technique_index in sorted(techniques_found):
            footnotes += "%d: %s\n" % (technique_index, Articles_annotations.techniques.get_technique(technique_index))

        return output_text, footnotes, legend


    def add_sentence_marker(self, line:str, row_counter:int)->str:

        if max(line.find("<span"), line.find("</span")) > -1: # there is an annotation in this row
            return '<div class="technique" id="row%d">%s</div>\n'%(row_counter, line)
        else:
            if len(line) <= 1: #empty line
                return '<br/>'
            else:   
                return '<div>%s</div>\n'%(line)


    def annotation_stack_index_to_markers_index(self, ind:int)->int:

        for x in range(len(self.markers)):
            if self.marker_annotation(x)==ind:
                return x
        sys.exit()


    def technique_index_from_annotation_index(self, x:int)->int:

        return self.techniques.indexOf(self.marker_label(self.annotation_stack_index_to_markers_index(x)))


    def start_annotation_marker_function(self, annotations_stack:list, marker_index:int, row_counter:int)->str:

        return '<span id="row%dannotation%d" class="%s">' \
                %(row_counter, self.marker_annotation(marker_index), " ".join([ "technique%d"%(self.technique_index_from_annotation_index(x)) for x in annotations_stack + [ self.marker_annotation(marker_index) ] ]))
        #if len(annotations_stack) > 0: # there is at least another tag opened, this one will overlap with it
        #    return '<span id="row%dannotation%d" class="%s overlappingtechniques">' \
        #        %(row_counter, self.marker_annotation(marker_index), " ".join([ "technique%d"%(self.technique_index_from_annotation_index(x)) for x in annotations_stack + [ self.marker_annotation(marker_index) ] ]))
        #else:
        #    return '<span id="row%dannotation%d" class="technique%d technique">'%(row_counter, self.marker_annotation(marker_index), techniques.indexOf(self.marker_label(marker_index))) 


    def end_annotation_marker_function(self, annotations_stack:list, marker_index:int, row_counter:int)->str:

        if self.marker_annotation() != annotations_stack[-1]: # we are facing this case: <t1> <t2> </t1> </t2> and we are about to close </t1> (self.marker_annotation()==</t1>, annotations_stack[-1]==</t2>), however, that case above is not allowed in HTML, therefore we are about to transform it to <t1> <t2> </t2></t1><t2> </t2> below
            new_annotations_stack = annotations_stack[ annotations_stack.index(self.marker_annotation()): ]
            res = "".join([ "</span>" for x in new_annotations_stack ]) # closing all tags opened that are supposed to continue after </t2>
            new_annotations_stack.remove(self.marker_annotation()) # removing </t1> from annotations_stack copy 
            technique_index = self.techniques.indexOf(self.marker_label(marker_index))
            res += '<sup id="row%dannotation%d" class="technique%d">%d</sup>'%(row_counter, self.marker_annotation(), technique_index, technique_index)
            for x in new_annotations_stack:
                new_annotations_stack.remove(x) # self.start_annotation_marker_function() assumes x is not in the annotations_stack variable passed as parameter
                res += self.start_annotation_marker_function(new_annotations_stack, self.annotation_stack_index_to_markers_index(x), row_counter) 
            return res
        else: # end of non-overlapping technique
            technique_index = self.techniques.indexOf(self.marker_label(marker_index))
            return '</span><sup id="row%dannotation%d" class="technique%d">%d</sup>'%(row_counter, self.marker_annotation(), technique_index, technique_index) 


    def tag_text_with_annotations(self, original_text, print_line_numbers=False):
        """
        mark the string original_text with object's annotations

        original_text: string with the text to be marked
        print_line_numbers: add line numbers to the text

        :return output_text the text in string original_text with added marks
                footnotes the list of techniques in the text
                legend description of the marks added
        """

        if Articles_annotations.techniques is None:
            if ans.Annotation.propaganda_techniques is None:
                Articles_annotations.techniques = Propaganda_Techniques()
            else:   
                Articles_annotations.techniques = ans.Annotation.propaganda_techniques

        self.get_markers_from_spans()

        output_text, curr_output_text_index, self.curr_marker = ("", 0, 0)
        techniques_found = set()
        row_counter = 1
        #print(self.markers)
        annotations_stack = []  # to handle overlapping annotations when assigning color background
        while curr_output_text_index < len(original_text):
            if self.curr_marker >= len(self.markers): # done marking text, need to flush the remaining content of <original_text> into <output_text>
                output_text += original_text[curr_output_text_index:]
                curr_output_text_index = len(original_text)
            else: # more markers have to be added to the content string
                if self.marker_position() <= curr_output_text_index: # it is time to add a marker
                    techniques_index = self.techniques.indexOf(self.marker_label())
                    techniques_found.add(techniques_index)
                    if self.is_starting_marker():
                        output_text += self.start_annotation_marker_function(annotations_stack, self.curr_marker, row_counter)
                        annotations_stack.append(self.marker_annotation())
                    else: 
                        output_text += self.end_annotation_marker_function(annotations_stack, self.curr_marker, row_counter)
                        annotations_stack.remove(self.marker_annotation())
                    self.curr_marker += 1
                else: # flush string content up to the next marker
                    text_to_be_added = original_text[curr_output_text_index:self.marker_position()]
                    row_counter += text_to_be_added.count('\n')
                    output_text += text_to_be_added
                    curr_output_text_index = self.marker_position()

        final_text = ""
        for row_counter, line in enumerate(output_text.split("\n"), 1):
            final_text += self.add_sentence_marker(line, row_counter)

        footnotes = "\n<div>List of techniques found in the article</div>\n\n"
        for technique_index in sorted(techniques_found):
            footnotes += "<div>%d: %s</div>\n" % (technique_index, self.techniques.get_technique(technique_index))

        return final_text, footnotes


    def merge_article_annotations(self, annotations_without_overlapping, i):
        """
        Checks if annotations_without_overlapping
        :param annotations_without_overlapping: a list of Annotations objects of an article assumed to be
                without overlapping
        :param i: the index in spans which needs to be tested for overlapping
        :return: 
        """
        #print("checking element %d of %d"%(i, len(spans)))
        if i<0:
            return True
        for j in range(0, i): #len(annotations_without_overlapping)):
            assert i<len(annotations_without_overlapping) or print(i, len(annotations_without_overlapping))
            if j != i and annotations_without_overlapping[i].span_overlapping(annotations_without_overlapping[j]):
                #   len(annotations_without_overlapping[i][1].intersection(annotations_without_overlapping[j][1])) > 0:
                # print("Found overlapping spans: %d-%d and %d-%d in annotations %d,%d:\n%s"
                #       %(min(annotations_without_overlapping[i][1]), max(annotations_without_overlapping[i][1]),
                #         min(annotations_without_overlapping[j][1]), max(annotations_without_overlapping[j][1]), i,j,
                #         print_annotations(annotations_without_overlapping)))
                annotations_without_overlapping[j].merge_spans(annotations_without_overlapping[i])
                #annotations_without_overlapping[j][1] = annotations_without_overlapping[j][1].union(annotations_without_overlapping[i][1])
                del(annotations_without_overlapping[i])
                # print("Annotations after deletion:\n%s"%(print_annotations(annotations_without_overlapping)))
                if j > i:
                    j -= 1
                # print("calling recursively")
                self.merge_article_annotations(annotations_without_overlapping, j)
                # print("done")
                return True

        return False


    def get_spans_content(self, article_content:str)->str: 
        """
        Given an article content as a string, prints the spans covered by the current annotations
        """
        s = ""
        for annotation in self.get_article_annotations():
            s += article_content[annotation.get_start_offset():annotation.get_end_offset()] + "\n"
        return s


    def remove_annotation(self, annotation_to_be_deleted:anwol.AnnotationWithOutLabel):

        self.get_article_annotations().remove(annotation_to_be_deleted)


    def remove_empty_annotations(self):

        self.spans = [ span for span in self.spans if span is not None ]


    def set_output_format(self, article_id=True, span=True, label=True):
        """
        Defines which fields are printed when annotations are written to standard output or file
        """
        self.output_format_article_id = article_id
        self.output_format_article_spans = span
        self.output_format_article_label = label


    def annotations_to_string_csv(self):
        """
        write article annotations, one per line, in the following format:
        article_id  label   span_start  span_end
        """
        span_string=""
        for span in self.spans:
            span_data = []
            if self.output_format_article_id:
                span_data.append(self.get_article_id())
            if self.output_format_article_label:
                span_data.append(span.get_label())
            if self.output_format_article_spans:
                span_data.append("%d\t%d"%(span.get_start_offset(), span.get_end_offset()))
            span_string += "\t".join(span_data) + "\n"

        return span_string


    def reset_annotations(self):

        self.spans = []


    @classmethod
    def set_start_annotation_effect(cls, new_effect:str)->None:

        cls.start_annotation_effect = new_effect


    @classmethod
    def set_end_annotation_effect(cls, new_effect:str)->None:

        cls.end_annotation_effect = new_effect


    @classmethod
    def set_start_annotation_str(cls, new_effect:str)->None:

        cls.start_annotation_str = new_effect


    @classmethod
    def set_end_annotation_str(cls, new_effect:str)->None:

        cls.end_annotation_str = new_effect


    @classmethod
    def set_annotation_background_color(cls, new_effect:str)->None:

        cls.annotation_background_color = new_effect


    def save_annotations_to_file(self, filename):
        
        with open(filename, "w") as f:
            f.write(self.annotations_to_string_csv())


    def shift_spans(self, start_index, offset):

        for span in self.spans:
            if span.get_start_offset() >= start_index:
                span.shift_annotation(offset)


    def sort_spans(self):
        """
        sort the list of annotations with respect to the starting offset
        """
        self.spans = sorted(self.spans, key=lambda span: span.get_start_offset() )


    def swap_annotations(self, i:int, j:int)->None:
        """
        Swap the i-th and j-th annotations 
        """
        self.spans[i], self.spans[j] = self.spans[j], self.spans[i]

