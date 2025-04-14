import os
import random
from PIL import Image
from utils import DatasetBase
template = ['a photo of {}, a type of food.']

class Food101Dataset(DatasetBase):
    def __init__(self, root_path, split='train', num_shots=0, transform=None, val_ratio=0.2, seed=42):
        self.root_path = root_path
        self.image_dir = os.path.join(self.root_path, "images")
        self.meta_dir = os.path.join(self.root_path, "meta")
        self.transform = transform
        self.template = template

        # Load class list
        
        #self._classnames = os.listdir(self.image_dir)
        # self._lab2cname, self._classnames = self.get_lab2cname(full_train)
        # self._cname2labels = self._cname2labels = {v: k for k, v in self._lab2cname.items()}
        self._classnames = sorted([
    d for d in os.listdir(self.image_dir)
    if os.path.isdir(os.path.join(self.image_dir, d)) and not d.startswith('.')
])
        self._cname2labels = {name: idx for idx, name in enumerate(self._classnames)}
        self._lab2cname = {idx: name for name, idx in self._cname2labels.items()}
        full_train = self.read_split(os.path.join(self.meta_dir, "train.txt"))
        train_items, val_items = self.split_trainval(full_train, val_ratio, seed)
        
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
            self.items = self.read_split(os.path.join(self.meta_dir, "test.txt"))
            self.test = self.items
            self.train = self.read_split(os.path.join(self.meta_dir, "train.txt"))
        train_data_for_stats = self.train if split in ['train', 'val'] else full_train
        super().__init__(root_path=root_path, shots=num_shots,
                         train_x=train_data_for_stats,
                         val=self.val if split == 'val' else [],
                         test=self.test if split == 'test' else [])

    def read_split(self, file_path):
        items = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                classname, img_name = line.strip().split("/")
                label = self._cname2labels[classname]
                impath = os.path.join(self.image_dir, classname, img_name + ".jpg")
                items.append((impath, label, classname))
        return items

    def split_trainval(self, items, val_ratio, seed):
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

if __name__ == '__main__':
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    
    # Set the dataset root directory.
    root_path = "../data/food-101"
    
    # Define image transforms.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset instances for each split.
    train_dataset = Food101Dataset(root_path, split='train', transform=transform, num_shots=4)
    print("num classes", train_dataset.num_classes)
    print("classnames", train_dataset.classnames)
    print("lab2cname", train_dataset.lab2cname)
    print("cname2lab", train_dataset._cname2labels)
    val_dataset = Food101Dataset(root_path, split='val', transform=transform)
    test_dataset = Food101Dataset(root_path, split='test', transform=transform)
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