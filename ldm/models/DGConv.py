import torch
import torch.nn as nn
import torch.nn.functional as F

# 随机选择中心像素的IDG
class IrregularDirectionalGradientConv(nn.Module):
    def __init__(self, kernel_size=3):
        super(IrregularDirectionalGradientConv, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, x):
        unfolded = F.unfold(x, self.kernel_size, padding=self.padding)
        center_idx = torch.randint(0, unfolded.size(2), (1,)).item()  # 随机选择中心像素索引
        center = unfolded[:, :, center_idx]
        surrounding = unfolded - center.unsqueeze(2)
        gradients = surrounding.mean(dim=2, keepdim=True)
        return gradients.view(x.size())

# 中心-周围梯度卷积
class CenterSurroundingGradientConv(nn.Module):
    def __init__(self, kernel_size=3):
        super(CenterSurroundingGradientConv, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, x):
        unfolded = F.unfold(x, self.kernel_size, padding=self.padding)
        center = unfolded[:, :, 4]  # 默认中心像素为位置4
        surrounding = unfolded.mean(dim=2)
        gradient = center - surrounding
        return gradient.view(x.size())

# 垂直梯度卷积
class VerticalGradientConv(nn.Module):
    def __init__(self, kernel_size=3):
        super(VerticalGradientConv, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, x):
        unfolded = F.unfold(x, self.kernel_size, padding=self.padding)
        gradient = unfolded[:, :, 3] - unfolded[:, :, 5]
        return gradient.view(x.size())

# 水平梯度卷积
class HorizontalGradientConv(nn.Module):
    def __init__(self, kernel_size=3):
        super(HorizontalGradientConv, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, x):
        unfolded = F.unfold(x, self.kernel_size, padding=self.padding)
        gradient = unfolded[:, :, 1] - unfolded[:, :, 7]
        return gradient.view(x.size())

# 自适应方向梯度卷积模块
class DGConvModule(nn.Module):
    def __init__(self, kernel_size=3):
        super(DGConvModule, self).__init__()
        self.idg = IrregularDirectionalGradientConv(kernel_size)
        self.csg = CenterSurroundingGradientConv(kernel_size)
        self.hg = HorizontalGradientConv(kernel_size)
        self.vg = VerticalGradientConv(kernel_size)

        # 自适应融合的可学习权重
        self.alpha_idg = nn.Parameter(torch.ones(1))
        self.alpha_csg = nn.Parameter(torch.ones(1))
        self.alpha_hg = nn.Parameter(torch.ones(1))
        self.alpha_vg = nn.Parameter(torch.ones(1))

        # 用于等效参数融合的线性层
        self.linear = nn.Linear(4, 1)  # 4个卷积输出的融合调整

    def forward(self, x):
        # 获取每个方向的梯度输出
        idg_out = self.idg(x)
        csg_out = self.csg(x)
        hg_out = self.hg(x)
        vg_out = self.vg(x)

        # 通过线性层融合这些卷积的输出
        fusion_out = torch.cat([idg_out, csg_out, hg_out, vg_out], dim=1)
        out = self.linear(fusion_out)  # 使用线性层调整输出

        # 自适应融合梯度
        out = (self.alpha_idg * idg_out +
               self.alpha_csg * csg_out +
               self.alpha_hg * hg_out +
               self.alpha_vg * vg_out)

        return out

# # 测试代码
# if __name__ == "__main__":
#     # 假设输入图像的形状是(batch_size, channels, height, width)
#     input_image = torch.randn(1, 3, 64, 64)  # 示例输入

#     # 初始化 DGConv 模块
#     model = DGConvModule(kernel_size=3)
#     output = model(input_image)

#     print("输出图像的形状:", output.shape)
