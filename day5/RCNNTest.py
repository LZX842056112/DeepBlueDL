# rcnn_selective_search_full.py
import cv2
import matplotlib.pyplot as plt
import selectivesearch.selectivesearch as ss
import numpy as np

def selective_search_rcnn_mode(image_path="image.jpg", num_regions=2000):
    """
    按照R-CNN模式生成大量候选框并展示

    Args:
        image_path: 图像文件路径
        num_regions: 要展示的候选区域数量
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法加载图像 {image_path}")
        return

    # 转换颜色空间用于显示
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 使用selective search提取更多候选区域
    # 调整参数以获得更多区域
    img_lbl, regions = ss.selective_search(
        img_rgb, scale=200, sigma=0.8, min_size=5
    )

    # 收集候选区域（不过滤，尽可能多地保留）
    candidates = []
    for r in regions:
        # 检查边界框是否有效
        x, y, w, h = r['rect']
        if x + w <= img.shape[1] and y + h <= img.shape[0] and w > 0 and h > 0:
            candidates.append((x, y, w, h, r['size']))

    # 按区域大小排序，优先保留较大的区域
    candidates.sort(key=lambda x: x[4], reverse=True)

    # 限制候选区域数量
    candidates = candidates[:num_regions]

    print(f"共找到 {len(candidates)} 个候选区域")

    # 创建可视化图像
    vis_img = img_rgb.copy()

    # 为不同区域使用不同颜色和透明度
    colors = [
        (0, 255, 0),    # 绿色
        (255, 0, 0),    # 蓝色
        (0, 0, 255),    # 红色
        (255, 255, 0),  # 青色
        (255, 0, 255),  # 紫色
        (0, 255, 255),  # 黄色
    ]

    # 绘制候选区域
    for i, (x, y, w, h, size) in enumerate(candidates):
        # 根据索引选择颜色
        color = colors[i % len(colors)]
        # 调整线宽基于区域大小
        thickness = max(1, min(3, int(size / 5000)))

        cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, thickness)

    # 使用matplotlib显示结果
    plt.figure(figsize=(15, 10))

    # 显示原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('原始图像')
    plt.axis('off')

    # 显示带候选框的图像
    plt.subplot(1, 2, 2)
    plt.imshow(vis_img)
    plt.title(f'R-CNN模式候选区域 (共{len(candidates)}个)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 分批展示候选区域（每批展示100个）
    batch_size = 100
    num_batches = min(5, (len(candidates) + batch_size - 1) // batch_size)  # 最多展示5批

    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, len(candidates))

        batch_img = img_rgb.copy()
        batch_candidates = candidates[start_idx:end_idx]

        for i, (x, y, w, h, size) in enumerate(batch_candidates):
            color = colors[i % len(colors)]
            cv2.rectangle(batch_img, (x, y), (x + w, y + h), color, 2)
            # 添加序号
            cv2.putText(batch_img, str(start_idx + i + 1), (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        plt.figure(figsize=(12, 8))
        plt.imshow(batch_img)
        plt.title(f'候选区域批次 {batch + 1} (编号 {start_idx + 1} - {end_idx})')
        plt.axis('off')
        plt.show()

        print(f"批次 {batch + 1}: 显示编号 {start_idx + 1} 到 {end_idx} 的候选区域")

    # 统计信息
    sizes = [r[4] for r in candidates]
    print(f"\n候选区域统计信息:")
    print(f"  总数: {len(candidates)}")
    print(f"  平均大小: {np.mean(sizes):.0f}")
    print(f"  最大大小: {np.max(sizes)}")
    print(f"  最小大小: {np.min(sizes)}")

# 运行演示
if __name__ == "__main__":
    selective_search_rcnn_mode("image.jpg", 2000)
# selective_search_demo.py
import cv2
import matplotlib.pyplot as plt
import selectivesearch.selectivesearch as ss
import numpy as np

def selective_search_demo(image_path="image.jpg"):
    """
    使用selective search算法检测图像中的候选区域并可视化
    
    Args:
        image_path: 图像文件路径
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法加载图像 {image_path}")
        return
    
    # 转换颜色空间用于显示
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 使用selective search提取候选区域
    img_lbl, regions = ss.selective_search(
        img_rgb, scale=500, sigma=0.9, min_size=10
    )
    
    # 过滤候选区域
    candidates = set()
    for r in regions:
        # 排除重复区域
        if r['rect'] in candidates:
            continue
        # 排除过小区域
        if r['size'] < 2000:
            continue
        # 检查边界框是否有效
        x, y, w, h = r['rect']
        if x + w > img.shape[1] or y + h > img.shape[0]:
            continue
        candidates.add(r['rect'])
    
    candidates = list(candidates)
    print(f"共找到 {len(candidates)} 个候选区域")
    
    # 创建可视化图像
    vis_img = img_rgb.copy()
    
    # 绘制所有候选区域
    for x, y, w, h in candidates:
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
    
    # 使用matplotlib显示结果
    plt.figure(figsize=(12, 8))
    
    # 显示原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('原始图像')
    plt.axis('off')
    
    # 显示带候选框的图像
    plt.subplot(1, 2, 2)
    plt.imshow(vis_img)
    plt.title(f'Selective Search候选区域 (共{len(candidates)}个)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 显示前20个最有可能的候选区域（更清晰的可视化）
    vis_img_top20 = img_rgb.copy()
    for i, (x, y, w, h) in enumerate(candidates[:20]):
        cv2.rectangle(vis_img_top20, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 添加序号
        cv2.putText(vis_img_top20, str(i+1), (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(vis_img_top20)
    plt.title('前20个候选区域')
    plt.axis('off')
    plt.show()

# 运行演示
if __name__ == "__main__":
    selective_search_demo("image.jpg")
