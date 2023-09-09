import torch
import torch.nn as nn

class SpecNet(nn.Module):
    """
    :param num_nodes: Number of graph node.
    :param num_features: Input features number.
    :param num_out: Output features number
    :param num_eigen: The number of eigenvector used.
    """
    
    def __init__(self, num_nodes, num_features, num_out, num_eigen, use_bias = False):
        super(SpecNet, self).__init__()
        
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_out = num_out
        self.num_eigen = num_eigen
        
        self.G = nn.ParameterList()
        for i in range(num_features):
            for j in range(num_out):
                self.G.append(nn.Parameter(torch.diag(torch.randn(num_eigen))))
            
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(torch.zeros(size=(num_out,))))
        else:
            self.register_parameter('bias', None)

        self.initialize_weights()
    
    def initialize_weights(self):
        for weight in self.G:
            nn.init.xavier_uniform_(weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, V):
        result = []
        for j in range(self.num_out):
            temp = torch.zeros((x.shape[0], self.num_nodes))
            for i in range(self.num_features):
                temp += torch.mm(V, torch.mm(self.G[j*self.num_features + i], torch.mm(V.T, x[:, :, i].T))).T
            result.append(temp)
        
        # Stack the results along a new dimension (last dimension)
        result = torch.stack(result, dim=-1)
               
        return result


batch_size = 32
num_nodes = 100
num_features = 32
num_out = 64
num_eigen = 3

x = torch.randn(batch_size, num_nodes, num_features)
V = torch.randn(num_nodes, num_eigen)

model = SpecNet(num_nodes, num_features, num_out, num_eigen)
output = model(x, V)

# 定义一个损失函数（示例中使用平方误差损失）
loss_fn = torch.nn.MSELoss()

# 随机生成目标张量，用于计算损失
target = torch.randn(batch_size, num_nodes, num_out)

# 计算损失
loss = loss_fn(output, target)

# 执行反向传播
loss.backward()

# 检查模型参数的梯度
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Parameter name: {name}, Gradient norm: {param.grad.norm().item()}")


        
