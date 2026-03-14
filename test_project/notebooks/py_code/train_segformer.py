
# 读取配置文件
import json
with open('D:\\pyLearn\\test_project\\notebooks\\config\\split_building_config.json', 'r') as f:
    config = json.load(f)
train_data_path = config['data']['train_data_path']
train_labels_path = config['data']['train_labels_path']
test_data_path = config['data']['test_data_path']
test_labels_path = config['data']['test_labels_path']
encoder=config['model']['encoder']
encoder_weights=config['model']['encoder_weights']
channels=config['model']['channels']
classes=config['model']['classes'] 
# 强制设置为1类（二分类：背景 vs 建筑），因为使用DiceLoss binary模式
print(f"Forcing classes to {classes} for Binary Segmentation task with DiceLoss.")

batch_size=config['data']['batch_size']
train_epochs=config['train']['epochs']
train_learning_rate=config['train']['learning_rate']
train_loss_function=config['train']['loss_function']
train_optimizer=config['train']['optimizer']
activation=config['model']['activation']
type=config['model']['type']
# 导入必要的库
import os
from torch.utils.data import DataLoader


# 1. 根据配置文件构建模型
import segmentation_models_pytorch as smp

if type == 'Unet':
    model = smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=channels, classes=classes, activation=activation)
elif type == 'FPN':
    model = smp.FPN(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=channels, classes=classes, activation=activation)
elif type == 'Linknet':
    model = smp.Linknet(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=channels, classes=classes, activation=activation)
elif type == 'Segformer':
    model = smp.Segformer(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=channels, classes=classes, activation=activation)
elif type == 'PSPNet':
    model = smp.PSPNet(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=channels, classes=classes, activation=activation)
else:
    raise ValueError(f"Unsupported model type: {type}")

# 2. 加载数据集方法定义
# 创建应用于WHU building segmentation的dataset类
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class building_dataset_train(Dataset):
    """
    自定义数据集类，用于读取WHU建筑分割数据集
    Args:
        img_dir: 图像文件夹路径
        label_dir: 标签文件夹路径
        transforms: 数据预处理/增强方法 (注意：如果是torchvision transforms，通常只处理image)
    """
    def __init__(self, img_dir, label_dir, transforms=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transforms = transforms
        
        self.img_list = sorted([f for f in os.listdir(img_dir) if f.endswith(('.tif', '.png', '.jpg'))])
        self.label_list = sorted([f for f in os.listdir(label_dir) if f.endswith(('.tif', '.png', '.jpg'))])
        
        assert len(self.img_list) == len(self.label_list), f"Images count ({len(self.img_list)}) and labels count ({len(self.label_list)}) do not match!"

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        label_name = self.label_list[idx]
        
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, label_name)
        
        # 读取
        image = Image.open(img_path).convert("RGB") # 如果不是RGB图像，转换为RGB
        mask = Image.open(label_path) # 保持原始模式，通常为单通道索引图

        # 预处理（最基础操作应当为转化为tensor）
        # 注意：如果使用Resize/Crop等几何变换，必须保证image和label同步变换
        if self.transforms:
            image = self.transforms(image)
        
        # 将label转为Tensor
        # 如果label是PIL图像，需要转为numpy数组再转Tensor，避免自动归一化到[0,1]（如果使用ToTensor的话）
        # 分割标签通常需要LongTensor类型
        mask = np.array(mask)
        # 将mask归一化为0和1 (通常WHU数据集是0和255)
        mask = mask.astype(np.float32)
        mask = mask / 255.0
        mask[mask > 0.5] = 1.0
        mask[mask <= 0.5] = 0.0

        mask = torch.from_numpy(mask).long() 
        
        # 如果label有额外的维度（如[H, W, 1]）， squeeze掉通道维变成 [H, W]
        if len(mask.shape) == 3:
            mask = mask.squeeze(-1)

        return image, mask, img_name
    

# 1. 定义损失函数与迭代优化器
import torch.optim as optim
import torch.nn as nn

# 适合分割任务的损失函数，Dice损失（适用于二分类分割）
# 若为多分类分割，可能需要使用CrossEntropyLoss或者结合DiceLoss和CrossEntropyLoss的混合损失函数
if train_loss_function == 'DiceLoss':
    # mode='binary' 需要输出 shape (B, 1, H, W)，所以需要将 mask 增加维度
    # Logits: (B, 1, H, W)
    # Target: (B, H, W) -> unsqueeze -> (B, 1, H, W)
    criterion = smp.losses.DiceLoss(mode='binary', from_logits=True)  # 二分类分割，Segformer 直接输出 Logits
elif train_loss_function == 'CrossEntropyLoss':
    criterion = nn.CrossEntropyLoss()
else:
    print(f"Unsupported loss function specified: {train_loss_function}, defaulting to DiceLoss.")
    criterion = smp.losses.DiceLoss(mode='binary', from_logits=True)
# 优化器选择,根据配置文件选择Adam或SGD，默认使用Adam
if train_optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=train_learning_rate)
elif train_optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=train_learning_rate, momentum=0.9)
else:
    print(f"Unsupported optimizer specified: {train_optimizer}, defaulting to Adam.")
    optimizer = optim.Adam(model.parameters(), lr=train_learning_rate)

# 2. 定义训练函数
def train(model,data_loader,criterion,optimizer,device):
    model.train()
    total_loss = 0
    for batach_index,(images, labels, _) in enumerate(data_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Segformer 输出为 [B, classes, H, W]
        # 如果 classes=1，则 output 为 [B, 1, H, W]
        # Label 为 [B, H, W]，此时需要 unsqueeze 使得 shape 为 [B, 1, H, W]
        if labels.ndim == 3:
            labels = labels.unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        # 输出训练进度
        if (batach_index + 1) % 50 == 0:
            print(f"Batch {batach_index + 1}/{len(data_loader)}, Loss: {loss.item():.4f}")
    avg_loss = total_loss / len(data_loader)
    print(f"Total Average training loss: {avg_loss:.4f}")


# 训练模型
# 这里的 train_data_path 应当遵照json中的配置，指向训练集的图像文件夹
train_img_dir = train_data_path
# train_labels_path 在配置文件中通常直接指向 label 文件夹
train_label_dir = train_labels_path 
data_transforms = transforms.Compose([
    transforms.ToTensor()
])

if __name__ == "__main__":
    # 设置环境变量以离线运行 (一旦权重下载完成)
    # 如果已经下载过权重，开启此选项可以跳过连接 Hugging Face，直接使用本地缓存
   os.environ['HF_HUB_OFFLINE'] = '1' 
   # 实例化 Dataset
   train_dataset = building_dataset_train(train_img_dir, train_label_dir, transforms=data_transforms)
   # 实例化 DataLoader
   train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=4)
   
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model.to(device)
   num_epochs = train_epochs
   for epoch in range(num_epochs):
       print(f"Epoch {epoch + 1}/{num_epochs}")
       train(model, train_loader, criterion, optimizer, device)