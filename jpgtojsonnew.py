import os
import json
import numpy as np
from PIL import Image, ImageEnhance
import cv2
from unet import Unet  # 假设unet.py在当前目录下

# 定义类别列表，需与训练时一致
classes = ["_background_", "z", "s", "w"]

def enhance_image(image, enhance_factor=1.2):
    """图像增强：亮度、对比度、锐度"""
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(enhance_factor)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(enhance_factor)
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(enhance_factor)

def generate_contours(pr, min_area=100, hole_filling=True):
    """生成轮廓并修复孔洞"""
    all_contours = []
    for class_id in np.unique(pr):
        if class_id == 0:
            continue
        # 为每个类别生成单独的掩码
        mask = np.zeros_like(pr)
        mask[pr == class_id] = 255
        
        # 填充孔洞
        if hole_filling:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cv2.drawContours(mask, [cnt], -1, 255, -1)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.extend([cnt for cnt in contours if cv2.contourArea(cnt) >= min_area])
    
    return all_contours

def adjust_epsilon(area, complexity):
    """根据面积和复杂度调整epsilon值"""
    if area > 10000:
        if complexity > 10:
            return 0.02
        else:
            return 0.01
    else:
        if complexity > 10:
            return 0.015
        else:
            return 0.005

def generate_json(image_path, pr, output_path, classes, skip_background=True, simplify=True):
    """生成优化后的JSON标注"""
    img = Image.open(image_path)
    width, height = img.size
    annotation = {
        "version": "4.5.6",
        "imagePath": os.path.basename(image_path),
        "imageHeight": height,
        "imageWidth": width,
        "shapes": [],
        "flags": {},
        "imageData": None
    }
    
    # 生成轮廓并分类
    contours = generate_contours(pr, min_area=100)
    
    for cnt in contours:
        class_id = pr[int(cnt[0][0][1]), int(cnt[0][0][0])]  # 取轮廓内第一个像素的类别
        if skip_background and class_id == 0:
            continue
            
        class_name = classes[class_id]
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, closed=True)
        complexity = perimeter / area if area > 0 else 0
        epsilon = adjust_epsilon(area, complexity)
        
        if simplify:
            approx = cv2.approxPolyDP(cnt, epsilon * perimeter, closed=True)
            points = approx.reshape(-1, 2).tolist()
        else:
            points = cnt.reshape(-1, 2).tolist()
        
        if len(points) >= 3:
            annotation["shapes"].append({
                "label": class_name,
                "points": points,
                "shape_type": "polygon",
                "flags": {}
            })
    
    # 保存紧凑JSON
    with open(output_path, 'w') as f:
        json.dump(annotation, f, separators=(',', ':'))

def batch_process_images(image_folder, output_folder, model, classes, preheat=True):
    """带模型预热的批量处理"""
    if preheat:
        # 预热模型（避免首次推理耗时影响结果）
        dummy_img = Image.new('RGB', (256, 256))
        model.get_miou_png(dummy_img)
    
    for filename in os.listdir(image_folder):
        if not filename.lower().endswith(('.jpg', '.png')):
            continue
            
        image_path = os.path.join(image_folder, filename)
        try:
            # 图像增强
            image = Image.open(image_path)
            image = enhance_image(image)
            
            # 模型推理
            pr = model.get_miou_png(image)
            pr = np.array(pr) if isinstance(pr, Image.Image) else pr
            
            # 生成JSON
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.json")
            generate_json(image_path, pr, output_path, classes)
            
            print(f"Processed: {image_path} -> {output_path}")
            
        except Exception as e:
            print(f"Error: {image_path} - {str(e)}")

if __name__ == "__main__":
    # 模型配置（根据训练参数调整）
    model = Unet(
        model_path="logs/best_epoch_weights.pth",
        num_classes=4,
        input_shape=[512, 512],  # 确保与训练时一致
        backbone="vgg"
    )
    
    # 处理参数
    image_folder = r"J:\1100\as - 副本"
    output_folder = r"J:\1100\as - 副本"
    
    # 执行处理（带预热和增强）
    batch_process_images(image_folder, output_folder, model, classes, preheat=True)