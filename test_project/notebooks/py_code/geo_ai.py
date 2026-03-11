import os

import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal

def build_sentinel_model():
    # 1. 配置设备 (适配你的 5060 Ti)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 定义 SMP 模型
    # 针对哨兵2号 RGB 数据：in_channels=3
    # 建筑物提取通常为二分类：classes=1 (使用 Sigmoid 激活)
    
    model = smp.Segformer(
        encoder_name="efficientnet-b4",        # ResNet34 轻量且收敛快，适合初学者
        encoder_weights=None,  # 使用预训练权重
        
        in_channels=3,                  
        classes=4,                      
        activation='sigmoid'            
    )

    # 加载本地预训练权重
    local_weight_path = "/home/jujue/.cache/torch/hub/checkpoints/resnet34-333f7ec4.pth"
    if os.path.exists(local_weight_path):
        print(f"加载本地预训练权重: {local_weight_path}")
        state_dict = torch.load(local_weight_path, map_location='cpu', weights_only=False)
        model.encoder.load_state_dict(state_dict, strict=False)
    else:
        print(f"警告: 本地预训练权重未找到: {local_weight_path}. 模型将随机初始化。")

    # 加载权重后再移到目标设备
    model = model.to(device)
    return model, device

def preprocess_sentinel(image_rgb):
    """
    针对哨兵2号数据的预处理：
    哨兵数据通常是 uint16 (0-10000)，需要拉伸到 0-1 并转为 Float32
    """
    # 输入为 [C, H, W]，先转 float32 并归一化到 [0, 1]
    max_val = float(np.max(image_rgb))
    if max_val <= 0:
        max_val = 1.0
    img = np.clip(image_rgb.astype(np.float32) / max_val, 0, 1)

    # 2-98% 拉伸（逐通道，CHW）
    for i in range(3):
        ch = img[i, :, :]
        p2, p98 = np.percentile(ch, (2, 98))
        if p98 > p2:
            ch = np.clip((ch - p2) / (p98 - p2), 0, 1)
        img[i, :, :] = ch

    # 返回 NCHW 的 float32 Tensor，便于直接送入 PyTorch 模型
    return torch.from_numpy(img).unsqueeze(0).float()


def pad_tensor_to_multiple(input_tensor, multiple=32):
    _, _, h, w = input_tensor.shape
    new_h = (h // multiple + 1) * multiple if h % multiple != 0 else h
    new_w = (w // multiple + 1) * multiple if w % multiple != 0 else w

    pad_h = new_h - h
    pad_w = new_w - w

    padded_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode="constant", value=0)
    return padded_tensor, h, w


def extract_and_stretch_rgb(source_rgb, valid_mask, lower=2, upper=98):
    """先按掩膜提取目标区域，再对提取区域做百分位拉伸。"""
    extracted = np.expand_dims(valid_mask, axis=2).repeat(3, axis=2) * source_rgb
    stretched = extracted.copy()
    valid_pixels = valid_mask > 0

    if not np.any(valid_pixels):
        return extracted, stretched

    for channel_index in range(stretched.shape[2]):
        channel = stretched[:, :, channel_index]
        values = channel[valid_pixels]
        p_low, p_high = np.percentile(values, (lower, upper))
        if p_high > p_low:
            channel[valid_pixels] = np.clip((values - p_low) / (p_high - p_low), 0, 1)
        stretched[:, :, channel_index] = channel

    return extracted, stretched


def infer_tiled(model, device, rgb, tile_size=1024, overlap=128, threshold=0.5):
    """使用分块推理避免一次性处理整幅大图导致显存溢出。"""
    _, height, width = rgb.shape
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("tile_size 必须大于 overlap")

    prob_sum = np.zeros((height, width), dtype=np.float32)
    hit_count = np.zeros((height, width), dtype=np.float32)
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for y in range(0, height, stride):
            y1 = min(y + tile_size, height)
            y0 = max(0, y1 - tile_size)
            for x in range(0, width, stride):
                x1 = min(x + tile_size, width)
                x0 = max(0, x1 - tile_size)

                tile_rgb = rgb[:, y0:y1, x0:x1]
                input_tensor = preprocess_sentinel(tile_rgb).to(device, non_blocking=True)

                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    pred = model(input_tensor).squeeze().float().cpu().numpy()

                prob_sum[y0:y1, x0:x1] += pred
                hit_count[y0:y1, x0:x1] += 1.0

                del input_tensor
                if use_amp:
                    torch.cuda.empty_cache()

    prob_map = prob_sum / np.maximum(hit_count, 1.0)
    return (prob_map > threshold).astype(np.uint8)

# --- 执行模拟测试 ---
model, device = build_sentinel_model()
model.eval()

gdal.UseExceptions()
data_path=("/mnt/d/pyLearn/pyLearn/test_project/data/tif/"
        "S2B_MSIL2A_20230611T030529_N0509_R075_T49RFL_20230611T053545_10m_4bands_croped.tif")
with gdal.Open(data_path) as src:
     band=src.ReadAsArray()


rgb=band[[2,1,0],:,:] # 选择需要的波段，注意波段索引可能需要调整
class_names = ["Class 0", "Class 1", "Class 2", "Class 3"]
building_class_index = 0

# # 使用分块推理，避免整图推理导致显存溢出
# mask = infer_tiled(model, device, rgb, tile_size=1024, overlap=128, threshold=0.5)
# 使用原始数据进行推理，适用于小图或显存充足的情况
input_tensor = preprocess_sentinel(rgb)
input_tensor, original_h, original_w = pad_tensor_to_multiple(input_tensor, multiple=32)
input_tensor = input_tensor.to(device)


with torch.no_grad():
    prob_map = model(input_tensor).squeeze().float().cpu().numpy()
prob_map = prob_map[:, :original_h, :original_w]
class_map = np.argmax(prob_map, axis=0).astype(np.uint8)
class_masks = [(class_map == class_index).astype(np.uint8) for class_index in range(len(class_names))]
building_mask = class_masks[building_class_index]
# mask=model(rgb.unsqueeze(0).float().to(device)).squeeze().float().cpu().numpy()
rgb_img=preprocess_sentinel(rgb).squeeze().cpu().numpy().transpose(1, 2, 0)
plt.imshow(class_map, cmap='tab10', vmin=0, vmax=len(class_names) - 1)
plt.title("Predicted Class Map")
plt.axis('off')
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for class_index, axis in enumerate(axes.ravel()):
    axis.imshow(class_masks[class_index], cmap='gray')
    axis.set_title(class_names[class_index])
    axis.axis('off')
plt.tight_layout()
plt.show()

# 对四个类别分别做掩膜提取并拉伸显示
extracted_per_class = []
stretched_per_class = []
for class_index, class_mask in enumerate(class_masks):
    extracted, stretched = extract_and_stretch_rgb(
        rgb_img,
        class_mask,
        lower=2,
        upper=98,
    )
    extracted_per_class.append(extracted)
    stretched_per_class.append(stretched)

    non_zero_pixels = int(np.count_nonzero(class_mask))
    total_pixels = int(class_mask.size)
    coverage = (non_zero_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0
    print(f"{class_names[class_index]} 像素: {non_zero_pixels}/{total_pixels} ({coverage:.4f}%)")

building = extracted_per_class[building_class_index]
building_stretched = stretched_per_class[building_class_index]

if np.count_nonzero(building_mask) == 0:
    print("警告: Extracted Buildings 为全黑，当前类别没有检测到有效像素。")

print(f"building 像素范围: min={building.min():.6f}, max={building.max():.6f}")

# 同一窗口显示：原始图像 + 四个类别提取结果（拉伸后）
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes_flat = axes.ravel()
axes_flat[0].imshow(rgb_img)
axes_flat[0].set_title("Input RGB Image")
axes_flat[0].axis('off')

for class_index in range(len(class_names)):
    axes_flat[class_index + 1].imshow(stretched_per_class[class_index])
    axes_flat[class_index + 1].set_title(f"{class_names[class_index]} Extracted")
    axes_flat[class_index + 1].axis('off')

axes_flat[-1].axis('off')
plt.tight_layout()
plt.show()

print(f"分割完成！Class Map 尺寸: {class_map.shape}")