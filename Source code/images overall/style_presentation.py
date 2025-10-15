import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义图像加载函数
def image_loader(image_name):
    loader = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整大小
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=cnn_normalization_mean, std=cnn_normalization_std)  # 归一化
    ])
    
    image = Image.open(image_name)  # 打开图像
    image = loader(image).unsqueeze(0)  # 增加一个维度
    print("Image shape:", image.shape)  # 输出图像形状
    return image.to(device, torch.float)

# 定义图像显示函数
def imshow(tensor, title=None):
    if tensor is None:
        print("Warning: Attempting to display a None tensor.")
        return
    
    unloader = transforms.ToPILImage()  # 创建图像转换器
    image = tensor.cpu().clone()  # 克隆张量
    image = unloader(image.squeeze(0))  # 去掉多余维度
    plt.imshow(image)  # 显示图像
    if title is not None:
        plt.title(title)  # 显示标题
    plt.pause(0.001)  # 暂停以便显示

# 定义模型和归一化参数
cnn = models.resnet50(weights='DEFAULT').to(device).eval()  # 加载ResNet50模型并设置为评估模式
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)  # 归一化均值
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)  # 归一化标准差

# 定义损失计算函数
def get_loss_and_features(model, input_img, style_img, content_img):
    # 计算内容和风格损失
    content_loss = nn.MSELoss()
    style_loss = nn.MSELoss()
    
    # 特征提取
    content_features = model(content_img)
    style_features = model(style_img)
    input_features = model(input_img)
    
    # 计算内容损失
    content_loss_value = content_loss(input_features, content_features)
    
    # 计算风格损失
    style_loss_value = style_loss(input_features, style_features)
    
    # 总损失
    total_loss = content_loss_value + style_loss_value
    return total_loss

# 定义样式迁移函数
def run_style_transfer(cnn, normalization_mean, normalization_std, content_img, style_img,
                       num_steps=100, style_weight=1000000, content_weight=10):
    input_img = content_img.clone()  # 克隆内容图像以进行优化
    optimizer = optim.Adam([input_img.requires_grad_()], lr=0.01)  # 定义优化器

    for i in range(num_steps):
        def closure():
            optimizer.zero_grad()  # 清除梯度
            output = input_img  # 使用输入图像
            loss = get_loss_and_features(cnn, output, style_img, content_img)  # 计算损失
            loss.backward()  # 反向传播
            return loss

        optimizer.step(closure)

        # 在每个步骤中计算损失
        loss_value = closure().item()  # 获取损失值并转换为标量
        if i % 50 == 0:  # 每50步打印损失
            print(f'Step {i}, Loss: {loss_value}')
            imshow(input_img, title='Output Image at Step {}'.format(i))  # 显示输出图像
            plt.savefig(f'output_step_{i}.png')  # 保存中间结果图像

    return input_img  # 返回处理后的图像张量

# 主函数
if __name__ == "__main__":
    image_directory = "images/"
    style_img = image_loader(image_directory + "style.jpg")  # 加载样式图像
    content_img = image_loader(image_directory + "content.jpg")  # 加载内容图像

    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                 content_img, style_img)  # 运行样式迁移

    # 显示结果
    plt.figure()
    imshow(output, title='Final Output Image')  # 显示输出图像
    plt.savefig('final_output.png')  # 保存最终输出图像
    plt.show()
