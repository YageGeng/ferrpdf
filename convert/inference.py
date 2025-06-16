from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.engine.results import Boxes  # 添加 Boxes 类型导入
import torch

# 这样可以帮助IDE理解类型
model: YOLO = YOLO("models/yolov12s-doclaynet.pt")
pred: list[Results] = model("images/test-page-0.jpg")

print()

for i, result in enumerate(pred):
    print(f"\n图像 {i+1}:")

    # 安全地获取图像路径
    if hasattr(result, 'path'):
        print(f"  路径: {result.path}")

    # 安全地获取图像尺寸
    if hasattr(result, 'orig_shape'):
        print(f"  图像尺寸: {result.orig_shape}")

    # 类型标注：result.boxes 的详细类型信息
    boxes: Boxes | None = result.boxes  # Boxes类型或None

    # 检查是否有检测框
    if boxes is not None and len(boxes) > 0:
        print(f"  检测到 {len(boxes)} 个对象:")
        print("  " + "-" * 40)

        # boxes 是一个可迭代的 Boxes 对象
        # 每次迭代返回的不是单个 box，而是 boxes 对象本身的切片
        for j in range(len(boxes)):
            try:
                # 获取边界框坐标
                # boxes.xyxy: torch.Tensor, shape: [N, 4]
                xyxy_tensor: torch.Tensor = boxes.xyxy[j]  # shape: [4]
                x1, y1, x2, y2 = xyxy_tensor.cpu().numpy()  # numpy.ndarray[4]

                # 获取置信度
                # boxes.conf: torch.Tensor, shape: [N]
                conf_tensor: torch.Tensor = boxes.conf[j]  # shape: []（标量张量）
                confidence: float = conf_tensor.cpu().numpy().item()  # Python float

                # 获取类别
                # boxes.cls: torch.Tensor, shape: [N]
                cls_tensor: torch.Tensor = boxes.cls[j]  # shape: []（标量张量）
                class_id: int = int(cls_tensor.cpu().numpy())  # Python int

                # model.names: dict[int, str]
                class_name: str = model.names[class_id] if hasattr(model, 'names') else f"Class_{class_id}"

                print(f"    对象 {j+1}:")
                print(f"      类别: {class_name} (ID: {class_id})")
                print(f"      置信度: {confidence:.4f}")
                print(f"      边界框: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
                print(f"      宽度: {x2-x1:.1f}, 高度: {y2-y1:.1f}")
                print()
            except Exception as e:
                print(f"    处理对象 {j+1} 时出错: {e}")
                print(f"    boxes 类型: {type(boxes)}")
                print(f"    boxes 属性: {dir(boxes)}")
    else:
        print("  未检测到任何对象")

    # 显示推理速度信息
    if hasattr(result, 'speed') and result.speed:
        print("  推理速度:")
        for key, value in result.speed.items():
            print(f"    {key}: {value:.2f}ms")

print("=" * 80)
