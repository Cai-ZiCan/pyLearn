# 使用训练好的模型权重，在大图中进行测试

remote_data_path="D:\\pyLearn\\WHU_build\\split_data\\predict\\2016_train.tif"

# 读取模型配置
# 包括模型类型、编码器、权重、输入通道数、类别数、激活函数等
import json
with open('D:\\pyLearn\\pyLearn\\test_project\\notebooks\\config\\split_building_config.json', 'r') as f:
    config = json.load(f)

type=config['model']['type'] # 模型类型，例如 "Unet", "FPN", "SegFormer" 等
encoder=config['model']['encoder'] # 编码器名称，例如 "resnet34", "mit_b0" 等
encoder_weights=config['model']['encoder_weights'] # 编码器权重，例如 "imagenet" 或 "None"（字符串形式）
channels=config['model']['channels'] # 输入图像的通道数，通常为3（RGB）或4（RGBA），根据数据集实际情况设置
classes=config['model']['classes'] # 输出类别数，对于二分类分割通常设置为1（背景 vs 目标），对于多分类分割则设置为类别总数
activation_cfg = config['model']['activation'] # 激活函数配置，通常为 "sigmoid"（适用于二分类）或 "softmax"（适用于多分类），也可以设置为 "None" 或 None 来表示不使用激活函数（让损失函数处理）
activation = activation_cfg # 训练过程中通常不直接在模型输出上应用激活函数，而是让损失函数（如DiceLoss或CrossEntropyLoss）处理原始Logits输出，因此这里的activation设置为None
model_weight_pth=config['train']['weight_pth']
train_width=config['data']['data_size']['width']
train_height=config['data']['data_size']['height']
train_loss_function=config['train']['loss_function'] # 训练时使用的损失函数，例如 "DiceLoss", "CrossEntropyLoss" 等，影响激活函数的选择

# 读取原始大图数据
from osgeo import gdal
import torch
import numpy as np
import segmentation_models_pytorch as smp
with gdal.Open(remote_data_path) as dataset:
    # 读取图像数据
    image_data = dataset.ReadAsArray()  # 形状通常为 (C, H, W)
    image_shape = image_data.shape # 记录原始图像的形状，方便后续处理和调试

    # 将影像裁剪为训练数据的大小（例如512x512），
    # 原始遥感图像较大，可以使用滑动窗口或其他方法进行裁剪，此处使用滑动窗口
    # 重叠度设置为0.5，以减少最后识别结果中的边界效应
    overlap = 0.5 # 重叠度
    crop_size = (train_height, train_width) # 裁剪大小
    # 根据指定重叠度和裁剪大小计算步长
    step_size = (int(crop_size[0] * (1 - overlap)), int(crop_size[1] * (1 - overlap)))
    # 计算裁剪的起始坐标列表
    crop_coords = []
    for y in range(0, image_shape[1] - crop_size[0] + 1, step_size[0]):
        for x in range(0, image_shape[2] - crop_size[1] + 1, step_size[1]):
            crop_coords.append((x, y))
    # 如果最后的裁剪区域不足以覆盖整个图像，可以添加一个额外的空白裁剪区域来覆盖剩余部分
    if crop_coords[-1][0] + crop_size[1] < image_shape[2]:
        crop_coords.append((image_shape[2] - crop_size[1], crop_coords[-1][1]))
    if crop_coords[-1][1] + crop_size[0] < image_shape[1]:
        crop_coords.append((crop_coords[-1][0], image_shape[1] - crop_size[0]))
    
    # 将裁剪的图像块保存到列表中，方便后续进行模型预测
    cropped_images = []
    for x, y in crop_coords:
        cropped_images.append(image_data[:, y:y+crop_size[0], x:x+crop_size[1]])
    

    # 输出原始图像的形状，确认读取是否正确
    print(f"Original image shape: {image_data.shape}")

    # 配置模型
# 统一 print 一下当前配置的模型类型，方便调试
print(f"Model type from config: {type}")

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
# 加载训练好的模型权重
model.load_state_dict(torch.load(model_weight_pth))
# 将模型移动到GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 将影像整理为模型输入的格式，并进行预测
import torch
import torch.nn as nn

# 对cropped_images中的每个图像块进行预测
predictions = []
# 对于每个裁剪的图像块，进行预处理并输入模型进行预测
for idx, cropped_image in enumerate(cropped_images):
    tensor_data=torch.from_numpy(cropped_image).float() # 转为float类型的Tensor，适用于模型输入
    tensor_data = tensor_data.unsqueeze(0) # 添加batch维度，变为 [1, C, H, W]
    tensor_data = tensor_data.to(device) # 将数据移动到GPU（如果可用）
    with torch.no_grad():
        output = model(tensor_data) # 模型输出，形状通常为 [1, classes, H, W]
        # 根据配置文件中的激活函数设置来应用相应的激活函数,获得概率图
        probs = _apply_activation(output) 
        # 对于二分类分割，通常会得到 [1, 1, H, W] 的概率图，表示每个像素属于建筑的概率
        if probs.dim() == 4 and probs.size(1) > 1:
            # Multi-class logits -> use class-1 probability for binary mask
            probs = probs[:, 1, :, :]
        else:
            probs = probs.squeeze(1) # 如果是单通道输出，去掉通道维度，变为 [1, H, W]
        
        pred=(probs > 0.5).float() # 二分类分割的预测结果，形状为 [1, H, W]，值为0或1
        predictions.append(pred.cpu().numpy()) # 将预测结果移动到CPU并转为numpy数组，方便后续处理

# 将预测结果进行后处理，例如拼接成完整的分割图，保存为文件等
# 拼接
# 将所有预测块按坐标加权拼接（重叠区域取平均）
full_pred_sum = np.zeros((image_shape[1], image_shape[2]), dtype=np.float32)
full_pred_count = np.zeros((image_shape[1], image_shape[2]), dtype=np.float32)

for (x, y), pred in zip(crop_coords, predictions):
    pred_patch = np.squeeze(pred).astype(np.float32)  # [H, W]
    h, w = pred_patch.shape
    full_pred_sum[y:y+h, x:x+w] += pred_patch
    full_pred_count[y:y+h, x:x+w] += 1.0

# 避免除零
full_pred_count[full_pred_count == 0] = 1.0

# 重叠区域平均后再二值化
full_pred_prob = full_pred_sum / full_pred_count
stitched_mask = (full_pred_prob > 0.5).astype(np.uint8)

print(f"Stitched mask shape: {stitched_mask.shape}")
print(f"Foreground ratio: {stitched_mask.mean():.4f}")

# 对于预测结果进行可视化，输出两幅影像，左侧为原始影像，右侧为预测的分割结果
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# 原始影像（取RGB通道）
axes[0].imshow(image_data[:3, :, :].transpose(1, 2, 0).astype(np.uint8)) # 转为 [H, W, C] 格式并显示
axes[0].set_title("Original Image")
# 预测的分割结果
axes[1].imshow(stitched_mask, cmap='gray') # 显示二值分割结果
axes[1].set_title("Predicted Segmentation Mask")

# 保存可视化结果
plt.savefig("D:\\pyLearn\\WHU_build\\split_data\\predict\\result\\segmentation_result.png")
plt.show()