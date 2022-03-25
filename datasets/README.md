##### README file for the PTC corpus Version 2
​
Contents
​
1. About the Propaganda Techniques Corpus (PTC)
2. The tasks
3. Data format
4. Citation
5. Changes in Version 2


About the Propaganda Techniques Corpus (PTC)
--------------------------------------------
​
This is the README file for the Propaganda Techniques Corpus (PTC), used at SemEval 2020 task 11. PTC is a corpus of texts annotated with 18 fine-grained propaganda techniques.
​
PTC was manually annotated by six professional annotators (both unitized and labeled) considering the fololowing 18 propaganda techniques:
​
* Loaded Language
* Name Calling,Labeling
* Repetition
* Exaggeration,Minimization
* Doubt
* Appeal to fear-prejudice
* Flag-Waving
* Causal Oversimplification
* Slogans
* Appeal to Authority
* Black-and-White Fallacy
* Thought-terminating Cliches
* Whataboutism
* Reductio ad Hitlerum
* Red Herring
* Bandwagon
* Obfuscation,Intentional Vagueness,Confusion
* Straw Men
​
Given the relatively low frequency of some of the techniques in the corpus, we decided to merge similar underrepresented techniques into one superclass:
​
For example, "Bandwagon" and "Reductio ad Hitlerum" are combined into "Bandwagon,Reductio ad Hitlerum".
Similarly, "Straw Men", "Red Herring" and "Whataboutism" are merged into "Whataboutism,Straw_Men,Red_Herring"
​
We further eliminated "Obfuscation,Intentional Vagueness,Confusion" as there were only 11 annotated cases in the whole corpus. Those two changes aim to simplify subtask 2 (see below). 
​
Tasks
--------------------------------------------
Among the different tasks that the corpus can enable, SemEval 2020 Task 11 focuses on the following ones:
​
Subtask 1. Propaganda Identification.
Given a plain-text document, identify specific fragments that contain a propaganda technique. This is a binary sequence tagging task.
​
Subtask 2. Propaganda Technique Labeling.
Given a text fragment identified as propaganda and its document context, identify the propaganda technique at hand. This is a multi-class classification problem.
​
​
Data format
--------------------------------------------
​
The corpus includes one plain text file and two annotation files per document. The annotation files are tab separated. The annotation file for subtask 2 includes the following columns:
​
C1. Article ID
C2. The propaganda technique applied in the instance.
C3. The starting offset of the propaganda instance (inclusive)
C4. The ending offset of the propaganda instance (exclusive)
​
The annotation file for subtask 1 is identical, except that it does not include C2 above.
​
C1. Article ID
C2. The starting offset of the propaganda instance (inclusive)
C3. The ending offset of the propaganda instance (exclusive)
​
Notes:
​
- Different propaganda techniques can overlap, either fully or partially.
- The articles were retrieved with newspaper3k library. The title is
in the first row, followed by an empty row. The content of the article
starts from the third row, one sentence per line (sentence splitting was
performed automatically using teh NLTK sentence splitter).
- The training and the development partitions of PTC include 446 articles (350k tokens) from 48 news outlets.
​
​
Citation 
--------------------------------------------
​
Please cite the following publication when using the PTC corpus:
​
G. Da San Martino, S. Yu, A. Barrón-Cedeño, R. Petrov and P. Nakov, “Fine-Grained Analysis of Propaganda in News Articles”, to appear at Conference on Empirical Methods in Natural Language Processing (EMNLP 2019), Hong Kong, China, November 3-7, 2019.
​
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



Changes in version 2
--------------------------------------------

The annotations have been revised and modified as follows:

1) All annotations whose starting offset was beyond the lenght of the corresponding article were deleted (1 instance)
2) All annotations whose ending offset was beyond the length of the article were trimmed such that they would end at the end of the article (2 instances)
3) In order to move towards standardised annotation boundaries, 
   all annotations were trimmed in order not to start or end with any of the following characters: " ", "\n", ".", ",", ":", ";" (456 instances)
4) We checked overlapping annotations for inconsistencies (~270 annotations modified) 

