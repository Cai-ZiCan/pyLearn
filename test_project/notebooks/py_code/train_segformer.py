
# 读取配置文件
import json
with open('D:\\pyLearn\\pyLearn\\test_project\\notebooks\\config\\split_building_config.json', 'r') as f:
    config = json.load(f)
train_data_path = config['data']['train_data_path']
train_labels_path = config['data']['train_labels_path']
test_data_path = config['data']['test_data_path']
test_labels_path = config['data']['test_labels_path']
encoder=config['model']['encoder']
encoder_weights=config['model']['encoder_weights']
channels=config['model']['channels']
classes=config['model']['classes'] 
# 设置为1类（二分类：背景 vs 建筑），因为使用DiceLoss binary模式


train_num_workers = config['train']['num_workers']
test_num_workers = config['test']['num_workers']
batch_size=config['data']['batch_size']
train_epochs=config['train']['epochs']
train_learning_rate=config['train']['learning_rate']
train_loss_function=config['train']['loss_function']
train_optimizer=config['train']['optimizer']
activation_cfg = config['model']['activation']
activation = activation_cfg
weight_pth=config['train']['weight_pth']
type=config['model']['type']
# 导入必要的库
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

# 1. 根据配置文件构建模型
import segmentation_models_pytorch as smp

# 统一 print 一下当前配置的模型类型，方便调试
print(f"Model type from config: {type}")
train_with_logits = train_loss_function in ['DiceLoss', 'CrossEntropyLoss']
if train_with_logits:
    # For training, keep raw logits and let the loss handle activation.
    print(f"对于训练，使用原始Logits输出，损失函数将自动处理激活,\
          设置的激活函数为: {activation}，仅在测试阶段应用")
    activation = None
if encoder_weights=="None":
    encoder_weights=None
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

class building_dataset(Dataset):
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

        if (train_loss_function == 'CrossEntropyLoss'):
            # CrossEntropyLoss 需要 LongTensor 类型的标签，且标签值为类别索引（0, 1, ...）
            mask = torch.from_numpy(mask).long()
        elif (train_loss_function == 'DiceLoss'):
            mask = torch.from_numpy(mask).float() # 转为float类型，适用于DiceLoss的binary模式
        
        # 如果mask有额外的通道维度（例如 [H, W, 1]），则squeeze掉变成 [H, W]
        # 因为对于DiceLoss的binary模式，标签应该是 [B, H, W] 或 [B, 1, H, W]，而不是 [B, H, W, 1]
        # 但是对于CrossEntropyLoss，标签应该是 [B, H, W]
        # 后续根据损失函数的要求，再在训练函数中进行适当的unsqueeze（例如对DiceLoss的binary模式）
        if len(mask.shape) == 3:
            mask = mask.squeeze(-1)

        return image, mask, img_name
    

# 1. 定义损失函数与迭代优化器
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# 适合分割任务的损失函数，Dice损失（适用于二分类分割）
# 若为多分类分割，可能需要使用CrossEntropyLoss或者结合DiceLoss和CrossEntropyLoss的混合损失函数
if train_loss_function == 'DiceLoss':
    # mode='binary' 需要输出 shape (B, 1, H, W)，所以需要将 mask 增加维度
    # Logits: (B, 1, H, W)
    # Target: (B, H, W) -> unsqueeze -> (B, 1, H, W)
    criterion = smp.losses.DiceLoss(mode='binary', from_logits=True)  # 二分类分割，使用原始Logits
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
    # loop是一个迭代器，使用tqdm包装后会显示进度条,
    # 其实际上对于训练而言相当于data_loader的一个增强版本，提供了更友好的训练进度显示和实时更新的功能
    # 故在使用for对data_loader进行迭代时，需要使用loop来替代原来的data_loader，才能使得loop生效显示进度条和相关信息
    # 使用total显示指定训练数据的总数（以batch为单位，以确保可以对进度条进行计算更新），
    # desc显示当前阶段（训练/测试），postfix显示当前batch的损失和显存占用
    loop = tqdm(data_loader, total=len(data_loader), desc=f"training")
    for batach_index,(images, labels, _) in enumerate(loop):
        images = images.to(device)
        labels = labels.to(device)
        if train_loss_function == 'DiceLoss':
            labels = labels.unsqueeze(1).float()
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        # 输出训练进度
        
         # 获取当前显存占用 (单位: GB)
        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated() / 1024**3 
            mem_info = f'{mem_used:.2f}GB'
        else:
            mem_info = 'CPU'
        # 使用tqdm显示当前batch的损失和显存占用，以及训练进度
        # 进度条会根据训练的进程自动更新，显示当前的batch索引和总batch数，以及当前batch的损失和显存占用
        # 百分比和剩余时间会根据训练的速度自动计算和显示，帮助你更好地了解训练的进度和资源使用情况  
        loop.set_postfix(batch_loss=loss.item(), gpu_mem=mem_info)
        # 输出每50个batch的损失
        if (batach_index + 1) % 50 == 0:
            print(f"Batch {batach_index + 1}/{len(data_loader)}, Loss: {loss.item():.4f}")
        
    avg_loss = total_loss / len(data_loader)
    print(f"Total Average training loss: {avg_loss:.4f}")

# 3. 定义测试函数（可选，后续可以在训练过程中或训练结束后调用）
def test(model,data_loader,criterion,device):
    model.eval()
    total_loss = 0
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    # 定义一个内部函数，根据配置文件中的激活函数设置来应用相应的激活函数
    def _apply_activation(outputs):
        if activation_cfg is None or activation_cfg == "None":
            if train_loss_function == 'DiceLoss':
                return torch.sigmoid(outputs)
            if train_loss_function == 'CrossEntropyLoss':
                return torch.softmax(outputs, dim=1)
            return outputs
        if activation_cfg == 'sigmoid':
            return torch.sigmoid(outputs)
        if activation_cfg == 'softmax':
            return torch.softmax(outputs, dim=1)
        return outputs
    with torch.no_grad():
        for images, labels, _ in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            if train_loss_function == 'DiceLoss':
                labels = labels.unsqueeze(1).float()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            probs = _apply_activation(outputs)
            if probs.dim() == 4 and probs.size(1) > 1:
                # Multi-class logits -> use class-1 probability for binary mask
                probs = probs[:, 1, :, :]
            else:
                probs = probs.squeeze(1)
            preds = (probs > 0.5).float()
            if labels.dim() == 4:
                # 如果标签有额外的通道维度，squeeze掉变成 [B, H, W]
                label_bin = labels.squeeze(1)
            else:
                label_bin = labels.float()

            total_tp += torch.sum((preds == 1) & (label_bin == 1)).item()
            total_fp += torch.sum((preds == 1) & (label_bin == 0)).item()
            total_fn += torch.sum((preds == 0) & (label_bin == 1)).item()
            total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    eps = 1e-7
    precision = total_tp / (total_tp + total_fp + eps)
    recall = total_tp / (total_tp + total_fn + eps)
    iou = total_tp / (total_tp + total_fp + total_fn + eps)
    print(f"Total Average testing loss: {avg_loss:.4f}")
    print(f"Test IoU: {iou:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

# 4. 训练模型
# 这里的 train_data_path 应当遵照json中的配置，指向训练集的图像文件夹
train_img_dir = train_data_path
# train_labels_path 在配置文件中通常直接指向 label 文件夹
train_label_dir = train_labels_path 
data_transforms = transforms.Compose([
    transforms.ToTensor()
])
    

def train_function():
   # 实例化 Dataset
   train_dataset = building_dataset(train_img_dir, train_label_dir, transforms=data_transforms)
   # 实例化 DataLoader
   train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=train_num_workers)
   
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model.to(device)
   num_epochs = train_epochs
   for epoch in range(num_epochs):
       print(f"Epoch {epoch + 1}/{num_epochs}")
       train(model, train_loader, criterion, optimizer, device)
    
   # 训练完成后，可以保存模型权重
   
   torch.save(model.state_dict(), weight_pth)

# 5. 测试模型（可选）
def test_function():
    # 实例化 Dataset
    test_dataset = building_dataset(test_data_path, test_labels_path, transforms=data_transforms)
    # 实例化 DataLoader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=test_num_workers)
    # 加载训练好的模型权重
    model.load_state_dict(torch.load(weight_pth))
    # 将模型移动到GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    test(model, test_loader, criterion, device)

if __name__ == "__main__":
    # 设置环境变量以离线运行 (一旦权重下载完成)
    os.environ['HF_HUB_OFFLINE'] = '1' 
    selection = int(input("Enter 0 to start training, 1 to start testing: "))
    if selection == 0:
       print("Starting training...")
       train_function()
    # 训练完成后，可以调用测试函数评估模型性能
    elif selection == 1:
       print("Starting testing...") 
       test_function()