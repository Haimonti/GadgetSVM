"""
GADGET SVM baseline — Gossip-bAseD sub-GradiEnT SVM.

Reference: "A Consensus Algorithm for Linear Support Vector Machines"
           (manuscript MS-0001-1922.65, Management Science submission)

Algorithm: Pegasos SGD + Push-Sum gossip across distributed sites.
Each site trains a local primal SVM model and gossips its weight vector
with one randomly chosen neighbour per iteration.
"""


class GADGETBaseline:
    pass
