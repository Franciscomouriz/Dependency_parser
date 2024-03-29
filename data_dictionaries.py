upos_dictionary = {"ADJ": 1, "ADP": 2, "ADV": 3, 
                   "AUX": 4, "CCONJ": 5, "DET": 6, 
                   "INTJ": 7, "NOUN": 8, "NUM": 9, 
                   "PART": 10, "PRON": 11, "PROPN": 12, 
                   "PUNCT": 13, "SCONJ": 14, "SYM": 15, 
                   "VERB": 16, "X": 17, "MASK": 0, "_": 0}

deprel_dictionary = {"None": 1, "acl": 2, "acl:relcl":3,
                     "advcl": 4, "advcl:relcl": 5, "advmod": 6,
                     "advmod:emph": 7, "amod": 8, "amod:emph": 9,
                     "advmod:lmod": 10, "amod": 11, "appos": 12,
                     "aux": 13, "aux:pass": 14, "case": 15,
                     "cc": 16, "cc:preconj": 17, "ccomp": 18,
                     "clf": 19, "compound": 20, "compound:lvc": 21,
                     "compound:prt": 22, "compound:redup": 23, "compound:svc": 24,
                     "conj": 25, "cop": 26, "csubj": 27,
                     "csubj:outer": 28, "csubj:pass": 29, "dep": 30,
                     "det": 31, "det:numgov": 32, "det:nummod": 33,
                     "det:poss": 34, "discourse": 35, "dislocated": 36,
                     "expl": 37, "expl:impers": 38, "expl:pass": 39,
                     "expl:pv": 40, "fixed": 41, "flat": 42,
                     "flat:foreign": 43, "flat:name": 44, "goeswith": 45,
                     "iobj": 46, "list": 47, "mark": 48,
                     "nmod": 49, "nmod:poss": 50, "nmod:tmod": 51,
                     "nsubj": 52, "nsubj:outer": 53, "nsubj:pass": 54,
                     "nummod": 55, "nummod:gov": 56, "obj": 57,
                     "obl": 58, "obl:agent": 59, "obl:arg": 60,
                     "obl:lmod": 61, "obl:tmod": 62, "orphan": 63,
                     "parataxis": 64, "punct": 65, "reparandum": 66,
                     "root": 67, "vocative": 68, "xcomp": 69, 
                     "det:predet": 70, "nmod:npmod":71}

arcs_dictionary = {"SHIFT": 1, "REDUCE": 2, "RIGHT-ARC": 3, "LEFT-ARC": 4, "END":5 }