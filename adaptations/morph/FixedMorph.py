import torch


class FixedMorph:

    def __init__(self, input_size, device):
        self.input_size = input_size  # (1024, 512)
        self.device = device

        self.affine_grid = self._affine_grid().to(device)
        self.affine_grid_inverse = self._affine_grid().to(device)

    def _affine_grid(self):
        H, W = self.input_size
        # 生成标准化坐标网格
        xx = torch.linspace(-1, 1, H)
        yy = torch.linspace(-1, 1, W)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')
        grid = torch.stack((grid_y, grid_x), -1)

        grid = grid.unsqueeze(0)
        flow = grid.permute(0, 3, 1, 2)

        flow.requires_grad = False
        return flow.clone().detach()
