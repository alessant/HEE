from collections import defaultdict
from typing import Iterable

import hypernetx as hnx

def read_data(filepath: str,
              feature_names: Iterable[str], 
              to_exclude: Iterable[str],
              sep=","):
    v2hes = {}
    hes2v = defaultdict(lambda: [])

    with open(filepath, "r") as f:
        for line in f:

            vals = line.split(sep)
            hes = []
      
            for index, elem in enumerate(vals):
                he = None

                if feature_names[index] in to_exclude:
                    continue

                # int(elem) == 1 and hes.append(feature_names[index]) 
                # int(elem) > 1 and hes.append(feature_names[index] + "_" + elem)

                if feature_names[index] == "type":
                    he = feature_names[index] + "_" + elem
                    
                elif int(elem) >= 1: 
                    he = feature_names[index]

                if int(elem) > 1:
                    he = feature_names[index] + "_" + elem

                if he is not None:
                    hes.append(he)
                    hes2v[he].append(vals[0])


            v2hes[vals[0]] = hes 
            

    h = hnx.Hypergraph(hes2v)

    # for k, v in v2hes.items():
    #     print(k, " ", v)

    # print("---")

    # for k, v in hes2v.items():
    #     print(k, " ", v)

    return h, v2hes, hes2v

