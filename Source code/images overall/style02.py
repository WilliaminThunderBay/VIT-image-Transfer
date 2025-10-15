import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import copy

# 图像加载和预处理
def image_loader(image_name, device):
    loader = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_name).convert("RGB")  # 确保图像是RGB格式
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# 反归一化函数
def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).to(tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).to(tensor.device).view(3, 1, 1)
    return tensor * std + mean

# 显示图像
def imshow(tensor, title=None):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = denormalize(image.squeeze(0))  # 对输出图像进行反归一化
    image = torch.clamp(image, 0, 1)  # 将像素值限制在[0,1]范围内
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

# 定义内容损失
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, x):
        self.loss = F.mse_loss(x, self.target)
        return x

# 定义风格损失
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()

    def forward(self, x):
        G = self.gram_matrix(x)
        self.loss = F.mse_loss(G, self.target)
        return x

    def gram_matrix(self, x):
        a, b, c, d = x.size()
        features = x.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

# 获取VGG模型及其损失层
def get_style_model_and_losses(cnn, style_img, content_img):
    cnn = copy.deepcopy(cnn)
    content_losses = []
    style_losses = []
    model = nn.Sequential()

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')
        
        model.add_module(name, layer)

        if name == 'conv_4':
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in {'conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'}:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    return model, style_losses, content_losses

# 样式迁移的核心函数
def run_style_transfer(cnn, content_img, style_img, input_img, num_steps=300, style_weight=1000000, content_weight=1000):
    print('Building the style transfer model...')
    model, style_losses, content_losses = get_style_model_and_losses(cnn, style_img, content_img)

    optimizer = optim.Adam([input_img.requires_grad_()], lr=0.01)  # 使用Adam优化器

    print('Optimizing...')
    for step in range(num_steps):
        def closure():
            optimizer.zero_grad()
            model(input_img)

            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            loss = style_weight * style_score + content_weight * content_score
            loss.backward()

            if step % 50 == 0:
                print(f"Step {step}: Style Loss: {style_score.item():.6f}, Content Loss: {content_score.item():.6f}")

            input_img.data.clamp_(0, 1)  # 确保输入图像数据在[0, 1]范围内
            return loss

        optimizer.step(closure)

    return input_img

# 主程序入口
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_img = image_loader("images/3.jpg", device)  # 替换为内容图片路径
    style_img = image_loader("images/4.jpg", device)  # 替换为风格图片路径

    input_img = content_img.clone()

    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    output = run_style_transfer(cnn, content_img, style_img, input_img)

    plt.figure()
    imshow(output, title='Output Image')
    plt.show()
