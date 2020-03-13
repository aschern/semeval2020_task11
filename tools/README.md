
Scorers for the Propaganda Techniques Corpus Version 2

Contents

1. Tasks
2. Evaluation scripts
3. Data format
4. Tools
5. Citation 
6. Changes from version 1 


Tasks
--------------------------------------------
The Propaganda Techniques Corpus (PTC) is a corpus of articles annotated 
with propaganda techniques at a fine-grained level. The list of 
techniques is in file data/propaganda-techniques-names-semeval2020task11.txt.
Among the different tasks that the corpus enables SemEval 2020 Task 11 focuses on the following ones:

Subtask 1 (SI). Propaganda Identification.
Given a plain-text document, identify those specific fragments that contain one propaganda technique. This is a binary sequence tagging task.

Subtask 2 (TC). Propaganda Technique Labeling.
Given a text fragment identified as propaganda and its document context, identify the applied propaganda technique at hand. This is a multi-class classification problem.

See the paper in the section "Citation" for further details. 


Evaluation scripts
--------------------------------------------

-Task SI (task-SI_scorer.py)

The evaluation script computes a variant of precision, recall, and F-measure 
that takes into account partial overlaps between fragments (see 
http://propaganda.qcri.org/semeval2020-task11/data/propaganda_tasks_evaluation.pdf
for more details). 

The script can be run as follows:

python3 task-SI_scorer.py -s [prediction_file] -r [gold_folder] -m

Note that all files *.labels in [gold_folder] will be considered containing gold labels
As an example, we provide a "prediction_file" data/submission-task-SI.tsv 
and you can run it as follows:

===

$ python3 task-SI_scorer.py -s data/submission-task-SI.tsv -r data -m
2019-09-20 19:47:26,427 - INFO - Checking user submitted file
2019-09-20 19:47:26,429 - INFO - Scoring the submission with precision and recall method
2019-09-20 19:47:26,430 - INFO - Precision=1.929825/2=0.964912  Recall=1.947458/4=0.486864
2019-09-20 19:47:26,430 - INFO - F1=0.647181


===

The scorer for the TC task is task-TC_scorer.py. 
The scorer requires file data/propaganda-techniques-names-semeval2020task11.txt. 
Such file contains the list of techniques used for scoring. 
Adding and removing items from the list will affect the outcome of the scorer. 
It can be run as follows

python3 task-TC_scorer.py -s [prediction_file] -r [gold_file] -p data/propaganda-techniques-names-semeval2020task11.txt

For example:

$ python3 task-TC_scorer.py -s data/submission-task-TC.tsv -r data/article736757214.labels-task-TC -p data/propaganda-techniques-names-semeval2020task11.txt 2>/dev/null
2019-09-20 19:39:21,286 - INFO - Checking format: User Predictions -- Gold Annotations
2019-09-20 19:39:21,287 - INFO - OK: submission file format appears to be correct
2019-09-20 19:39:21,293 - INFO - Scoring submission
F1=0.600000
Precision=0.600000
Recall=0.600000
F1_Appeal_to_Authority=0.0
F1_Appeal_to_fear-prejudice=0.0
F1_Bandwagon,Reductio_ad_hitlerum=0.0
F1_Black-and-White_Fallacy=0.0
F1_Causal_Oversimplification=0.0
F1_Doubt=0.0
F1_Exaggeration,Minimisation=1.0
F1_Flag-Waving=0.0
F1_Loaded_Language=0.6666666666666666
F1_Name_Calling,Labeling=0.6666666666666666
F1_Repetition=0.0
F1_Slogans=0.0
F1_Thought-terminating_Cliches=0.0
F1_Whataboutism,Straw_Men,Red_Herring=0.0


Data format
--------------------------------------------

-Task SI

The corpus includes one tab-separated file per article in the following 
format: 

id   begin_offset     end_offset

where 
	id is the identifier of the article
	begin_offset is the character where the covered span begins (inclusive)
	end_offset is the character where the covered span ends (exclusive)

An example of such a file is data/article736757214.task-FLC.labels. 

-Task TC

The corpus includes one tab-separated file per article in the following format:

id   technique    begin_offset     end_offset

The fields are the same as for task SI, but it now also includes "technique", i.e., the propaganda technique applied in the instance. 


Tools
--------------------------------------------

- The script print_spans.py highlights the annotations in an article.

python3 print_spans.py -s [annotations_file] -t [article_file] -p [propaganda_techniques_file]

For example:

python3 print_spans.py -t data/article736757214.txt -s data/article736757214.labels-task-TC -p data/propaganda-techniques-names-semeval2020task11.txt


Citation 
--------------------------------------------

Please cite the following publication when using the PTC corpus:

G. Da San Martino, S. Yu, A. Barrón-Cedeño, R. Petrov and P. Nakov, "Fine-Grained Analysis of Propaganda in News Articles", to appear at the Conference on Empirical Methods in Natural Language Processing (EMNLP 2019), Hong Kong, China, November 3-7, 2019.

@InProceedings{EMNLP19DaSanMartino,
author = {Da San Martino, Giovanni and
Yu, Seunghak and
Barr\'{o}n-Cede\~no, Alberto and
Petrov, Rostislav and
Nakov, Preslav},
title = {Fine-Grained Analysis of Propaganda in News Articles},
booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7, 2019},
series = {EMNLP-IJCNLP 2019},
year = {2019},
address = {Hong Kong, China},
month = {November},
}


Changes from version 1
--------------------------------------------

Fixed a bug in the evaluation function for task TC that prevented to find the best alignment between the labels of identical spans in certain cases.

Now print_spans.py has a parameter -p specifying the file with the list of propaganda techniques

