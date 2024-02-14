# reference: https://github.com/JeongJiHeon/ScoreDiffusionModel/blob/main/NCSN/NCSN_example.ipynb
import torch
import torch.nn as nn
import math
import pdb


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layer):
        super().__init__()
        self.layers = nn.ModuleList( [nn.Linear(dim_in, dim_hidden)] + [nn.Linear(dim_hidden, dim_hidden) for _ in range(num_layer)] + [nn.Linear(dim_hidden, dim_out)] )
        self.gelu = nn.GELU()
    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx + 1 != len(self.layers):
                x = self.gelu(x)
        return x


class Model(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_layer, n_steps, sigma_min, sigma_max):
        '''
        Score Network.

        n_steps   : perturbation schedule steps (Langevin Dynamic step)
        sigma_min : sigma min of perturbation schedule
        sigma_min : sigma max of perturbation schedule
        '''
        super().__init__()
        self.sigmas = torch.exp(torch.linspace(start=math.log(sigma_max), end=math.log(sigma_min), steps = n_steps))

        '''
        self.linear_model1 = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.Dropout(p),
            nn.GELU(),
        )
        '''
        self.linear_model1 = MLP(dim_in, dim_hidden, dim_hidden, num_layer)

        # Condition sigmas
        self.embedding_layer = nn.Embedding(n_steps, dim_hidden)

        '''        
        self.linear_model2 = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.Dropout(p),
            nn.GELU(),
            
            nn.Linear(dim_hidden, dim_hidden),
            nn.Dropout(p),
            nn.GELU(),
            
            nn.Linear(dim_hidden, dim_in),
        )
        '''
        self.linear_model2 = MLP(dim_hidden, dim_hidden, dim_in, num_layer)

    def loss_fn(self, x, idx=None):
        '''
        This function performed when only training phase.

        x          : real data if idx==None else perturbation data
        idx        : if None (training phase), we perturbed random index. Else (inference phase), it is recommended that you specify.
        '''
        scores, target, sigma = self.forward(x, idx=idx, get_target=True)
        
        target = target.view(target.shape[0], -1)
        scores = scores.view(scores.shape[0], -1)

        '''
        print(scores)
        print(target)
        pdb.set_trace()
        '''

        losses = torch.square(scores - target).mean(dim=-1) * sigma.squeeze() ** 2
        return losses.mean(dim=0)
        
    def forward(self, x, idx=None, get_target=False):
        '''
        x          : real data if idx==None else perturbation data
        idx        : if None (training phase), we perturbed random index. Else (inference phase), it is recommended that you specify.
        get_target : if True (training phase), target and sigma is returned with output (score prediction)
        '''
        if idx == None:
            idx = torch.randint(0, len(self.sigmas), (x.size(0),)).to(x.device)
            used_sigmas = self.sigmas[idx][:,None].to(x.device)
            noise = torch.randn_like(x).to(x.device)
            x_tilde = x + noise * used_sigmas
            
        else:
            idx = torch.cat([torch.Tensor([idx for _ in range(x.size(0))])]).long().to(x.device)
            used_sigmas = self.sigmas[idx][:,None].to(x.device)
            x_tilde = x

        if get_target:
            target = - 1 / used_sigmas * noise

        output = self.linear_model1(x_tilde)
        embedding = self.embedding_layer(idx)
        output = self.linear_model2(output + embedding)

        return (output, target, used_sigmas) if get_target else output

