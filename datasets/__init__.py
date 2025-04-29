from datasets.oxfordpets import OxfordPetsDataset
from datasets.food101 import Food101Dataset
from datasets.eurosat import EuroSATDataset
from datasets.dtd import DTDTextureDataset
from datasets.caltech101 import Caltech101Dataset
dataset_lists = {
    'oxford_pets': OxfordPetsDataset ,
    'food-101': Food101Dataset,
    'eurosat' : EuroSATDataset,
    'dtd' : DTDTextureDataset,
    'caltech-101': Caltech101Dataset,
}
