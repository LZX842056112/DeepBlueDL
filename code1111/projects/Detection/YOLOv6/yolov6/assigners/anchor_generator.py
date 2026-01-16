import torch


def generate_anchors(feats, fpn_strides, grid_cell_size=5.0, grid_cell_offset=0.5, device='cpu', is_eval=False,
                     mode='af'):
    '''Generate anchors from features. 通用的生成anchor box的代码'''
    anchors = []
    anchor_points = []
    stride_tensor = []
    num_anchors_list = []
    assert feats is not None
    if is_eval:
        for i, stride in enumerate(fpn_strides):
            _, _, h, w = feats[i].shape
            shift_x = torch.arange(end=w, device=device) + grid_cell_offset
            shift_y = torch.arange(end=h, device=device) + grid_cell_offset
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            anchor_point = torch.stack(
                [shift_x, shift_y], axis=-1).to(torch.float)
            if mode == 'af':  # anchor-free
                anchor_points.append(anchor_point.reshape([-1, 2]))
                stride_tensor.append(
                    torch.full(
                        (h * w, 1), stride, dtype=torch.float, device=device))
            elif mode == 'ab':  # anchor-based
                anchor_points.append(anchor_point.reshape([-1, 2]).repeat(3, 1))
                stride_tensor.append(
                    torch.full(
                        (h * w, 1), stride, dtype=torch.float, device=device).repeat(3, 1))
        anchor_points = torch.cat(anchor_points)
        stride_tensor = torch.cat(stride_tensor)
        return anchor_points, stride_tensor
    else:
        for i, stride in enumerate(fpn_strides):  # 遍历各个输出层的feature map下采样的比例值
            _, _, h, w = feats[i].shape  # 获取得到当前feature map的大小
            cell_half_size = grid_cell_size * stride * 0.5  # 一般的边框高度&宽度
            shift_x = (torch.arange(end=w, device=device) + grid_cell_offset) * stride  # center_x
            shift_y = (torch.arange(end=h, device=device) + grid_cell_offset) * stride  # center_y
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            anchor = torch.stack(
                [
                    shift_x - cell_half_size, shift_y - cell_half_size,
                    shift_x + cell_half_size, shift_y + cell_half_size
                ],
                dim=-1
            ).clone().to(feats[0].dtype)  # [x1,y1,x2,y2]
            anchor_point = torch.stack([shift_x, shift_y], dim=-1).clone().to(feats[0].dtype)  # 中心点坐标

            if mode == 'af':  # anchor-free
                anchors.append(anchor.reshape([-1, 4]))
                anchor_points.append(anchor_point.reshape([-1, 2]))
            elif mode == 'ab':  # anchor-based
                anchors.append(anchor.reshape([-1, 4]).repeat(3, 1))
                anchor_points.append(anchor_point.reshape([-1, 2]).repeat(3, 1))
            num_anchors_list.append(len(anchors[-1]))
            stride_tensor.append(
                torch.full(
                    [num_anchors_list[-1], 1], stride, dtype=feats[0].dtype))
        anchors = torch.cat(anchors)
        anchor_points = torch.cat(anchor_points).to(device)
        stride_tensor = torch.cat(stride_tensor).to(device)
        return anchors, anchor_points, num_anchors_list, stride_tensor
