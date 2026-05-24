import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    
    if idx is None:
        idx = knn(x, k=k)
        
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature

class VectorFieldDGCNN(nn.Module):
    def __init__(self, k=20):
        super(VectorFieldDGCNN, self).__init__()
        self.k = k
        
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.decode1 = nn.Sequential(nn.Conv1d(1024 + 512, 512, 1, bias=False),
                                     nn.BatchNorm1d(512),
                                     nn.LeakyReLU(negative_slope=0.2))
        self.decode2 = nn.Sequential(nn.Conv1d(512, 256, 1, bias=False),
                                     nn.BatchNorm1d(256),
                                     nn.LeakyReLU(negative_slope=0.2))
        self.decode3 = nn.Conv1d(256, 3, 1)

    def forward(self, x):
        B, _, N = x.size()
        
        x1 = get_graph_feature(x, k=self.k)
        x1 = self.conv1(x1)
        x11 = x1.max(dim=-1, keepdim=False)[0]
        
        x2 = get_graph_feature(x11, k=self.k)
        x2 = self.conv2(x2)
        x22 = x2.max(dim=-1, keepdim=False)[0]
        
        x3 = get_graph_feature(x22, k=self.k)
        x3 = self.conv3(x3)
        x33 = x3.max(dim=-1, keepdim=False)[0]
        
        x4 = get_graph_feature(x33, k=self.k)
        x4 = self.conv4(x4)
        x44 = x4.max(dim=-1, keepdim=False)[0]
        
        local_features = torch.cat((x11, x22, x33, x44), dim=1)
        
        x5 = self.conv5(local_features)
        global_features = x5.max(dim=2, keepdim=True)[0]
        global_features = global_features.repeat(1, 1, N) 
        
        concat = torch.cat([local_features, global_features], dim=1)
        out = self.decode1(concat)
        out = self.decode2(out)
        displacement = self.decode3(out)
        
        return displacement