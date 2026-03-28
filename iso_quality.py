import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 配置路径
input_dir = "test_images"
output_dir = "output"

if not os.path.exists(input_dir):
    print("❌ 请创建 test_images 文件夹并放入图片")
    exit()

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 存储结果
names = []
sharp_list = []
noise_list = []

print("="*50)
print("ISP (sharpness + noise)")
print("="*50)

# 遍历处理
for fname in os.listdir(input_dir):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(input_dir, fname)
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️  跳过：{fname}")
        continue

    # 计算
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    sharpness = np.mean(edges)
    noise = np.std(gray)

    # 保存数据
    names.append(fname)
    sharp_list.append(sharpness)
    noise_list.append(noise)

    # 生成对比图
    compare_img = np.hstack((gray, edges))
    save_path = os.path.join(output_dir, f"对比_{fname}")
    cv2.imwrite(save_path, compare_img)

    # 打印
    print(f"\n{fname}")
    print(f"  清晰度:{sharpness:6.2f}")
    print(f"  噪声  :{noise:6.2f}")

plt.figure(figsize=(12,6))

# 画曲线 + 点
plt.plot(names, sharp_list, color='#ff4d4d', marker='o', linestyle='-', linewidth=2, markersize=7,
          label='sharpness')
plt.plot(names, noise_list, color='#3399ff', marker='s', linestyle='-', linewidth=2, markersize=7, 
         label='noise')

# 所有数值 统一显示在点的下方，不被覆盖
for i in range(len(names)):
    # 清晰度数值（往下偏移 3）
    plt.text(i, sharp_list[i] - 3, f"{sharp_list[i]:.1f}",
             color="#ff4d4d", ha="center", fontsize=10, weight='bold')
    
    # 噪声数值（往下偏移 3）
    plt.text(i, noise_list[i] - 3, f"{noise_list[i]:.1f}",
             color="#3399ff", ha="center", fontsize=10, weight='bold')

plt.title('ISP sharpness & noise curve comparison', fontsize=14)
plt.xticks(rotation=30)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(output_dir, "曲线图_清晰数值.png"), dpi=150)
plt.show()

print("\n" + "="*50)
print("✅ 全部完成！")
print(f"📁 对比图 + 曲线图 已保存到：{output_dir}")
print("="*50)