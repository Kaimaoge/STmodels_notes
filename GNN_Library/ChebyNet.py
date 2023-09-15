import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim

class ChebGraphConv(nn.Module):
    def __init__(self, K, in_features, out_features):
        super(ChebGraphConv, self).__init__()
        self.K = K
        self.weight = nn.Parameter(torch.FloatTensor(K, in_features, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x, gso):
        # Chebyshev polynomials:
        # x_0 = x,
        # x_1 = gso * x,
        # x_k = 2 * gso * x_{k-1} - x_{k-2},
        # where gso = 2 * gso / eigv_max - id.

        cheb_poly_feat = []
        if self.K < 0:
            raise ValueError('ERROR: The order of Chebyshev polynomials shoule be non-negative!')
        elif self.K == 0:
            # x_0 = x
            cheb_poly_feat.append(x)
        elif self.K == 1:
            # x_0 = x
            cheb_poly_feat.append(x)

                # x_1 = gso * x
            cheb_poly_feat.append(torch.einsum('bij,ik->bkj', x, gso))
        else:
            # x_0 = x
            cheb_poly_feat.append(x)
            cheb_poly_feat.append(torch.einsum('bij,ik->bkj', x, gso))
            for k in range(2, self.K):
                cheb_poly_feat.append(torch.einsum('bij,ik->bkj', cheb_poly_feat[k - 1], 2*gso)  - cheb_poly_feat[k - 2])

    
        feature = torch.stack(cheb_poly_feat, dim=-1)
        cheb_graph_conv = torch.einsum('bnik,kij->bnj', feature, self.weight)

        return cheb_graph_conv
