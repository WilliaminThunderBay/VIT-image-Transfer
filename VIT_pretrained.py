import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import timm  # 使用timm加载ViT模型
import copy

# 确认设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置图像尺寸为ViT预期的尺寸
imsize = 224

# 图像加载和预处理
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),  # 修改为固定大小 224x224
    transforms.ToTensor()
])

def image_loader(image_name):
    image = Image.open(image_name)
    image = image.resize((imsize, imsize))
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# 加载风格和内容图像
image_directory = "images/"
style_img = image_loader(image_directory + "style.jpg")
content_img = image_loader(image_directory + "content.jpg")

# 显示图像的辅助函数
unloader = transforms.ToPILImage()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

# 加载ViT模型
vit = timm.create_model('vit_base_patch16_224', pretrained=True).to(device).eval()

# 风格损失和内容损失的计算
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()

    def forward(self, input):
        G = self.gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

    def gram_matrix(self, input):
        batch_size, n_features, tokens = input.size()
        features = input.view(batch_size * n_features, tokens)
        G = torch.mm(features, features.t())
        return G.div(batch_size * n_features * tokens)

# 将风格和内容损失插入ViT模型
def get_style_model_and_losses(vit, style_img, content_img):
    vit = copy.deepcopy(vit)

    content_losses = []
    style_losses = []

    # ViT模型是一个 transformer encoder，选择适当的层进行特征提取
    model = nn.Sequential()
    i = 0

    for name, layer in vit.named_children():
        model.add_module(name, layer)
        if name == 'blocks':  # blocks是ViT中Transformer Encoder的核心部分
            for block in layer:
                i += 1
                model.add_module(f"block_{i}", block)

                # 对每个block的输出提取特征
                if i == 4:  # 选择第4层进行内容损失的计算（可以调整）
                    target = model(content_img).detach()
                    content_loss = ContentLoss(target)
                    model.add_module(f"content_loss_{i}", content_loss)
                    content_losses.append(content_loss)

                if i in [1, 3, 5, 7]:  # 选择多层进行风格损失的计算
                    target_feature = model(style_img).detach()
                    style_loss = StyleLoss(target_feature)
                    model.add_module(f"style_loss_{i}", style_loss)
                    style_losses.append(style_loss)

    return model, style_losses, content_losses

# 使用内容图像的副本作为输入图像
input_img = content_img.clone()

# 优化器
def get_input_optimizer(input_img):
    optimizer = optim.Adam([input_img.requires_grad_()])
    return optimizer

# 运行风格迁移
def run_style_transfer(vit, style_img, content_img, input_img, num_steps=3000,
                       style_weight=500000000, content_weight=1):
    model, style_losses, content_losses = get_style_model_and_losses(vit, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    for step in range(num_steps):
        def closure():
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            if step %500 == 0:
                print(f"Step {step}: Style Loss : {style_score.item()} Content Loss: {content_score.item()}")

                # 每应过滤函数步骤显示输出的图像
                plt.figure()
                imshow(input_img, title=f'Intermediate Output at Step {step}')

            return style_score + content_score

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)

    return input_img

# 运行风格迁移并显示输出
output = run_style_transfer(vit, style_img, content_img, input_img, num_steps=400)

plt.figure()
imshow(output, title='Output Image')
plt.show()
