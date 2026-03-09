import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import time

def test_smp_inference():
    # 1. 环境检查
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"检测到设备: {device} | 显卡: {torch.cuda.get_device_name(0)}")

    # 2. 初始化 SMP 模型 (以 Unet + ResNet50 为例)
    # 针对遥感：in_channels=4 (假设为 RGB + NIR), classes=1 (二分类，如建筑物)
    model = smp.Unet(
        encoder_name="resnet50",        
        encoder_weights="imagenet",     
        in_channels=4,                  
        classes=1,                      
        activation='sigmoid'
    ).to(device)
    
    model.eval()

    # 3. 模拟遥感大图切片数据 (Batch_size=4, Channels=4, Height=512, Width=512)
    dummy_input = torch.randn(4, 4, 512, 512).to(device)

    print("\n--- 开始性能测试 ---")
    
    # 4. 使用 5060 Ti 推荐的自动混合精度 (AMP) 进行推理
    # 这能显著降低显存占用并提升 Blackwell 架构的吞吐量
    try:
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                # 热身运行 (Warm-up)
                _ = model(dummy_input)
                
                # 正式计时运行
                start_time = time.time()
                for _ in range(50):
                    output = model(dummy_input)
                end_time = time.time()

        print(f"推理完成！输出尺寸: {output.shape}")
        print(f"50次迭代总耗时: {end_time - start_time:.4f} 秒")
        print(f"平均每帧 (Batch) 耗时: {(end_time - start_time)/50*1000:.2f} 毫秒")
        
    except Exception as e:
        print(f"推理过程中出错: {e}")

    # 5. 显存占用检查
    print(f"当前显存占用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

if __name__ == "__main__":
    test_smp_inference()