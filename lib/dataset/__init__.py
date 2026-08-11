from typing import Dict, Type
from .PASTISDataset import PASTISDataset
from .Sen2324Dataset import Sen2324Dataset
from .Sen2324Dataset_disjoint import Sen2324Datasetdj

DATASETS = {
    "pastis": PASTISDataset,
    "sen2324": Sen2324Dataset,
    "sen2324dj": Sen2324Datasetdj,
}
