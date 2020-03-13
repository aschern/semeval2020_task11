
class Propaganda_Techniques():


    TECHNIQUE_NAMES_FILE="data/propaganda-techniques-names.txt"

    def __init__(self, filename=TECHNIQUE_NAMES_FILE):

        with open(filename, "r") as f:
             self.techniques = [ line.rstrip() for line in f.readlines() ]


    def get_propaganda_techniques_list(self)->list:

        return self.techniques


    def get_propaganda_techniques_list_sorted(self)->list:

        return sorted(self.techniques)


    def is_valid_technique(self, technique_name):

        return technique_name in self.techniques


    def __str__(self):

        return "\n".join(self.techniques)


    def __getitem__(self, index):
        return self.techniques[index]


    def get_technique(self, index):
        return self.techniques[index]


    def indexOf(self, technique_name):
        return self.techniques.index(technique_name)
