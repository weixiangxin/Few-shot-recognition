import os
import sys
import h5py
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

# utils
# 定义图像预处理流程
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为224x224
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor()  # 将图像转换为Tensor格式
])

# 从指定文件夹加载所有图像
def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):  # 遍历文件夹中的每个文件
        img_path = os.path.join(folder_path, filename)  # 获取图像完整路径
        img = Image.open(img_path).convert("RGB")  # 打开图像并转换为RGB格式
        img = data_transforms(img)  # 应用预处理
        images.append(img)  # 将处理后的图像添加到列表中
    return images

# 加载单个图像并进行预处理
def load_image(img_path):
    img = Image.open(img_path).convert("RGB")  # 打开图像并转换为RGB格式
    img = data_transforms(img)  # 应用预处理
    return img

# 保存提取的特征到HDF5文件
def save_features(features, path):
    with h5py.File(path, "w") as f:  # 创建或打开HDF5文件
        for class_name, feature in features.items():
            f.create_dataset(class_name, data=feature)  # 为每个类别创建数据集并保存特征

# 从HDF5文件加载特征
def load_features(path):
    features = {}
    with h5py.File(path, "r") as f:  # 读取HDF5文件
        for class_name in f.keys():  # 遍历文件中的每个数据集
            features[class_name] = torch.tensor(f[class_name][...])  # 将特征转换为Tensor并存储
    return features

# models
# 定义特征提取模型
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # 从当前目录加载预训练的EfficientNet
        model_path = os.path.join(os.getcwd(), 'efficientnet-b2-8bb594d6.pth')
        self.feature_extractor = EfficientNet.from_name('efficientnet-b2')  # 创建模型
        self.feature_extractor.load_state_dict(torch.load(model_path))  # 加载权重

    def forward(self, x):
        features = self.feature_extractor.extract_features(x)  # 提取图像特征
        return features.view(features.size(0), -1)  # 展平特征以适应后续处理

# feature_extractor
# 提取支持集特征
def extract_support_features(support_dir, feature_save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有可用的GPU
    model = FeatureExtractor().to(device)  # 初始化特征提取模型并转移到设备
    model.eval()  # 设置模型为评估模式
    support_features = {}

    for class_name in os.listdir(support_dir):  # 遍历支持集中的每个类别
        class_path = os.path.join(support_dir, class_name)  # 获取类别路径
        if not os.path.isdir(class_path):  # 检查是否是文件夹
            continue

        images = load_images_from_folder(class_path)  # 加载该类别的图像
        features = []

        with torch.no_grad():  # 禁用梯度计算以节省内存
            for img in images:
                img = img.unsqueeze(0).to(device)  # 添加批次维度并转移到设备
                feature = model(img)  # 提取特征
                features.append(feature.squeeze(0))  # 移除批次维度并保存特征

        class_prototype = torch.stack(features).mean(dim=0)  # 计算类别原型特征
        support_features[class_name] = class_prototype.cpu().numpy()  # 转移到CPU并转换为NumPy数组

    save_features(support_features, feature_save_path)  # 保存提取的特征
    print(f"支持集特征已保存到 {feature_save_path}")

# classifier
# 分类查询图像
def classify_query_images(query_dir, support_features_path):
    support_features = load_features(support_features_path)  # 加载支持集特征
    model = FeatureExtractor()  # 初始化特征提取模型
    model.eval()  # 设置模型为评估模式

    predictions = []

    for img_name in os.listdir(query_dir):  # 遍历查询目录中的图像
        if not img_name.endswith('.png'):  # 只处理PNG文件
            continue

        img_path = os.path.join(query_dir, img_name)
        img = load_image(img_path)  # 加载图像

        with torch.no_grad():  # 禁用梯度计算
            query_feature = model(img.unsqueeze(0)).squeeze(0)  # 提取查询图像特征

        max_similarity = float('-inf')  # 初始化最大相似度
        predicted_class = None

        for class_name, class_prototype in support_features.items():
            # 使用余弦相似度计算特征之间的相似度
            similarity = F.cosine_similarity(query_feature, torch.tensor(class_prototype).to(query_feature.device), dim=0)
            if similarity > max_similarity:  # 更新最大相似度和预测类别
                max_similarity = similarity
                predicted_class = class_name

        predictions.append(img_name + ',' + predicted_class)  # 保存预测结果
        print(f"图像 {img_name} 分类为: {predicted_class}")

    return predictions

def main(to_pred_dir, result_save_path):
    run_py = os.path.abspath(__file__)
    model_dir = os.path.dirname(run_py)

    dirpath = os.path.abspath(to_pred_dir)
    filepath = os.path.join(dirpath, 'testA')
    task_lst = os.listdir(filepath)

    res = ['img_name,label']
    for task_name in task_lst:
        support_path = os.path.join(filepath, task_name, 'support')
        query_path = os.path.join(filepath, task_name, 'query')
        extract_support_features(support_path, "support_features.h5")
        res.extend(classify_query_images(query_path, "support_features.h5"))

    with open(result_save_path, 'w') as f:
        f.write('\n'.join(res))

if __name__ == "__main__":
    to_pred_dir = sys.argv[1]
    result_save_path = sys.argv[2]
    main(to_pred_dir, result_save_path)