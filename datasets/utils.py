import os
import random
from typing import List, Optional

class Datum:
    """
    A simple data structure to represent a sample.
    """
    def __init__(self, impath: str, label: int, classname: Optional[str] = None):
        self.impath = impath
        self.label = label
        self.classname = classname

    def __repr__(self):
        return f"Datum(impath={self.impath}, label={self.label}, classname={self.classname})"

class DatasetBase:
    """
    Base class for datasets with standard train/val/test split support.
    """
    def __init__(self, root_path=None, shots=0, train_x: Optional[List[Datum]] = None, val: Optional[List[Datum]] = None, test: Optional[List[Datum]] = None):
        self.root_path = root_path
        self.shots = shots
        self.train = train_x or []
        self.val = val or []
        self.test = test or []

        
        self._num_classes = self.get_num_classes(self.train)
        self._lab2cname, self._classnames = self.get_lab2cname(self.train)


        # self._cname2labels = self.get_cname2labels(train_x)
    
    def generate_fewshot_dataset(self, data, num_shots, seed=42):
        if num_shots <= 0:
            return data

        random.seed(seed)
        label_to_items = {}
        for item in data:
            # Try to get the label from an attribute; if not, assume tuple (impath, label, classname).
            try:
                label = item.label
            except AttributeError:
                label = item[1]
            label_to_items.setdefault(label, []).append(item)

        fewshot_data = []
        for label, items in label_to_items.items():
            # Randomly sample up to num_shots items for this label.
            sampled = random.sample(items, min(num_shots, len(items)))
            fewshot_data.extend(sampled)

        return fewshot_data

    def __len__(self):
        return len(self.train) + len(self.val) + len(self.test)

    def __repr__(self):
        return f"<DatasetBase: {len(self.train)} train, {len(self.val)} val, {len(self.test)} test>"
    
    def get_num_classes(self, data_source):
        """Count number of classes.

        Args:
            data_source (list): a list of Datum objects.
        """
        label_set = set()
        for item in data_source:
            try:
                label_set.add(item.label)
            except AttributeError:
                # If item is a tuple, assume label is at index 1.
                label_set.add(item[1])
        return max(label_set) + 1
    
    def get_lab2cname(self, data_source):
        """Get a label-to-classname mapping (dict).

        Args:
            data_source (list): a list of Datum objects.
        """
        container = set()
        for item in data_source:
            try:
                container.add((item.label, item.classname))
            except AttributeError:
                # If item is a tuple, assume label is at index 1.
                container.add((item[1], item[2]))
        mapping = {label: classname for label, classname in container}
        labels = list(mapping.keys())
        labels.sort()
        classnames = [mapping[label] for label in labels]
        return mapping, classnames
    
    # def get_cname2labels(self, data_source):
    #     """Get a label-to-classname mapping (dict).

    #     Args:
    #         data_source (list): a list of Datum objects.
    #     """
    #     container = set()
    #     for item in data_source:
    #         container.add((item.label, item.classname))
    #     mapping = {classname: label for label, classname in container}
    #     classnames = list(mapping.keys())
    #     classnames.sort()
    #     labels = [mapping[classname] for classname in classnames]
    #     return mapping
    

    
    @property
    def num_classes(self):
        return self._num_classes
    
    @property
    def classnames(self):
        return self._classnames
    
    @property
    def lab2cname(self):
        return self._lab2cname
    
    @property
    def cname2labels(self):
        return self._cname2labels