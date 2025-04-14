import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.utils import DatasetBase, Datum
import random
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image

# class OxfordPets(DatasetBase):
#     dataset_dir = "oxfordpets"

#     def __init__(self, root_path, shots=0, transform=None):
#         super().__init__(root_path=root_path, shots=shots)  # Initialize the base class
#         self.template = template
#         self.image_dir = os.path.join(self.root_path, "images")  # Directory where images are stored
#         self.annotation = os.path.join(self.root_path, "annotations")
        
#         items = self.read_split(os.path.join(self.annotation, "trainval.txt"), self.image_dir)
#         train, val = self.split_trainval(items)
#         test = self.read_split(os.path.join(self.annotation, "test.txt"), self.image_dir)
#         self.train = train
#         self.val = val
#         self.test = test
#         self.transform = transform
#         super().__init__(train_x=train, val=val, test=test)

#     #@staticmethod
#     def read_split(self, file_path, path_prefix):
#         """
#         Read a split from the annotation file.
#         """
#         with open(file_path, 'r') as f:
#             lines = f.readlines()

#         items = []
#         for line in lines:
#             parts = line.strip().split()
#             if len(parts) != 4:
#                 continue
#             img_name, label, _, _ = parts
#             img_name = img_name + ".jpg"
#             impath = os.path.join(path_prefix, img_name)
#             label = int(label) - 1  # Convert to 0-based index
#             classname = "_".join(img_name.split("_")[:-1]).lower()  # Extract breed/class name

#             item = Datum(
#                 impath=impath,
#                 label=label,
#                 classname=classname
#             )
#             items.append(item)
        
#         return items
    
#     def split_trainval(self, items, val_ratio=0.2, seed=42):
#         random.seed(seed)
#         random.shuffle(items)
#         total = len(items)
#         val_size = int(total * val_ratio)

#         val = items[:val_size]
#         train = items[val_size:]

#         return train, val
    
#     def __getitem__(self, index):
#         datum = self.train[index]
#         image = Image.open(datum.impath).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         return image, datum.classname

# transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std= [0.229, 0.224, 0.225])]
#     )

# if __name__ == '__main__':
#     dataset = OxfordPets("../data/oxfordpets", shots=2, transform=transform)
#     print(len(dataset.train), len(dataset.val), len(dataset.test))
#     train_loader = DataLoader(dataset.train, batch_size=32, shuffle=True, num_workers=4)
#     val_loader = DataLoader(dataset.val, batch_size=32, shuffle=True, num_workers=4)
#     test_loader = DataLoader(dataset.test, batch_size=32, shuffle=True, num_workers=4)
#     for images, labels in train_loader:
#             print("Batch images shape:", images.shape)  # e.g., (32, 3, 224, 224)
#             print("Batch labels shape:", labels.shape)  # e.g., (32,)
#             break

import os
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

template = ["a photo of a {}, a type of pet"]
# class OxfordPetsDataset(DatasetBase):
#     def __init__(self, root_path, split='train', num_shots = 0, transform=None, val_ratio=0.2, seed=42):
#         """
#         Args:
#             root_path (str): Root directory where the Oxford Pets dataset is stored.
#             split (str): Which split to load. Must be 'train', 'val', or 'test'.
#                          For training and validation, the annotation file 'trainval.txt' is used.
#                          For testing, the annotation file 'test.txt' is used.
#             transform: A torchvision.transforms object to preprocess images.
#             val_ratio (float): Ratio for the validation split from trainval.txt.
#             seed (int): Random seed for the train/val split.
#         """
#         self.root_path = root_path
#         self.image_dir = os.path.join(self.root_path, "images")
#         self.annotation_dir = os.path.join(self.root_path, "annotations")
#         self.transform = transform
#         self.template = template
#         if split in ['train', 'val']:
#             # Load full trainval annotations and split them.
#             items = self.read_split(os.path.join(self.annotation_dir, "trainval.txt"))
#             train_items, val_items = self.split_trainval(items, val_ratio, seed)
#             self.train = train_items
#             self.val = val_items
#             self.items = self.train if split == 'train' else val_items
#             self.train = self.generate_fewshot_dataset(self.items, num_shots) if split in ['train', 'val'] else []
#         elif split == 'test':
#             # Load test items
#             self.items = self.read_split(os.path.join(self.annotation_dir, "test.txt"))
#             self.test = self.items
#         else:
#             raise ValueError("split must be one of 'train', 'val', or 'test'")

        
#         # Initialize the base class with splits.
#         super().__init__(root_path=root_path, shots=seed,
#                          train_x=self.train if split in ['train', 'val'] else [],
#                          val=self.val if split in ['train', 'val'] else [],
#                          test=self.test if split == 'test' else [])

class OxfordPetsDataset(DatasetBase):
    def __init__(self, root_path, split='train', num_shots=0, transform=None, val_ratio=0.2, seed=42):
        self.root_path = root_path
        self.image_dir = os.path.join(self.root_path, "images")
        self.annotation_dir = os.path.join(self.root_path, "annotations")
        self.transform = transform
        self.template = template
        
        # Always load full trainval (for class information)
        full_trainval = self.read_split(os.path.join(self.annotation_dir, "trainval.txt"))
        train_items, val_items = self.split_trainval(full_trainval, val_ratio, seed)
        
        # Set splits for train/val/test
        if split in ['train', 'val']:
            self.train = train_items
            self.val = val_items
            self.items = self.train if split == 'train' else self.val
            # Apply few-shot sampling if specified.
            if num_shots > 0:
                self.train = self.generate_fewshot_dataset(self.train, num_shots)
                self.items = self.train
        elif split == 'test':
            test_items = self.read_split(os.path.join(self.annotation_dir, "test.txt"))
            desired_size = len(val_items)
            if len(test_items) > desired_size:
                random.seed(seed)
                test_items = random.sample(test_items, desired_size)
            self.items = test_items
            self.test = self.items
            # Use training split from trainval to compute number of classes
            self.train = train_items
        
        # Now, pass non-empty lists to the base class
        super().__init__(root_path=root_path, shots=seed,
                         train_x=self.train if self.train else [],
                         val=self.val if split in ['train', 'val'] else [],
                         test=self.test if split == 'test' else [])
 
    def read_split(self, file_path):
        """
        Reads an annotation file and returns a list of tuples:
        (impath, label, classname).
        Assumes each line in the file has four fields.
        """
        items = []
        with open(file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            img_name, label, _, _ = parts
            img_name = img_name + ".jpg"
            impath = os.path.join(self.image_dir, img_name)
            label = int(label) - 1 
            classname = "_".join(img_name.split("_")[:-1]).lower()
            items.append((impath, label, classname))
        return items
    
    def split_trainval(self, items, val_ratio, seed):
        """
        Shuffles and splits 'items' into training and validation sets.
        """
        random.seed(seed)
        random.shuffle(items)
        total = len(items)
        val_size = int(total * val_ratio)
        val_items = items[:val_size]
        train_items = items[val_size:]
        return train_items, val_items
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        impath, label, classname = self.items[index]
        image = Image.open(impath).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, classname



# Example usage:
if __name__ == '__main__':
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    
    # Set the dataset root directory.
    root_path = "../data/oxfordpets"
    
    # Define image transforms.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset instances for each split.
    train_dataset = OxfordPetsDataset(root_path, split='train', transform=transform, num_shots=4)
    print("num classes", train_dataset.num_classes)
    print("classnames", train_dataset.classnames)
    print("lab2cname", train_dataset.lab2cname)
    val_dataset = OxfordPetsDataset(root_path, split='val', transform=transform)
    test_dataset = OxfordPetsDataset(root_path, split='test', transform=transform)
    print("classname", val_dataset._num_classes)
    print("classname", test_dataset._num_classes)
    print("Train set size:", len(train_dataset))
    print("Validation set size:", len(val_dataset))
    print("Test set size:", len(test_dataset))
    
    # Create DataLoaders.
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4)
    
    # Example: iterate over one batch from each DataLoader.
    for images, labels, classnames in train_loader:
        print("Train Batch - images shape:", images.shape)
        print("Train Batch - labels:", labels)
        print("Train Batch - classnames:", classnames)
        break