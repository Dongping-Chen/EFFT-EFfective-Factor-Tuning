from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import os
from scipy.io import loadmat

class CustomImageNetValDataset(Dataset):
    def __init__(self, img_folder, ground_truth_file, transform=None):
        self.img_folder = img_folder
        self.transform = transform
        with open(ground_truth_file, 'r') as f:
            self.labels = [int(line.strip()) for line in f.readlines()]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_name = f"ILSVRC2012_val_{str(index + 1).zfill(8)}.JPEG"
        img_path = os.path.join(self.img_folder, img_name)
        image = Image.open(img_path).convert('RGB')
        label = self.labels[index] - 1  # 将标签从1-based转为0-based

        if self.transform:
            image = self.transform(image)

        return image, label
    
class CustomImageNetTrainFolder(ImageFolder):
    def __init__(self, root, folder_to_label_mapping, transform=None, target_transform=None):
        self.folder_to_label_mapping = folder_to_label_mapping
        super(CustomImageNetTrainFolder, self).__init__(root, transform=transform, target_transform=target_transform)

    def __getitem__(self, index):
        """
        Overrides the default method to use the provided mapping for labels.
        """
        # Use the original method to get the image and "incorrect" label
        sample, target = super(CustomImageNetTrainFolder, self).__getitem__(index)
        # Correct the target using the provided mapping
        class_name = self.classes[target]
        correct_target = self.folder_to_label_mapping[class_name]
        return sample, correct_target
    
def Imagenet_val_loader(batch_size=64):
    val_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_dataset = CustomImageNetValDataset(img_folder='./Imagenet/val', 
                                       ground_truth_file='./Imagenet/ILSVRC2012_validation_ground_truth.txt',
                                       transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return val_loader


def Imagenet_train_loader(batch_size=64):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    meta = loadmat('./Imagenet/meta.mat')
    # 从.mat文件中提取'synsets'数组
    synsets = meta['synsets']
    # 初始化空字典
    id_to_wnid = {}
    wnid_to_id = {}

    # 遍历'synsets'数组
    for entry in synsets:
        # 提取'ILSVRC2012_ID'和'WNID'
        ilsvrc_id = entry[0][0][0][0]
        WNID = entry[0][1][0]
        if int(ilsvrc_id) > 1000:
            break
        id_to_wnid[int(ilsvrc_id) - 1] = WNID
        wnid_to_id[WNID] = int(ilsvrc_id) - 1

    train_dataset = CustomImageNetTrainFolder(root='./Imagenet/train', folder_to_label_mapping = wnid_to_id, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 2. 检查数据加载
    print("\nSample data from the dataset:")
    for i, (image, label) in enumerate(train_dataset):
        # 为了演示，只打印前5个样本的路径和标签
        if i >= 5:
            break
        print(f"Path: {train_dataset.samples[i][0]}, Label: {label}")
    
    return train_loader