# reference: https://github.com/JeongJiHeon/ScoreDiffusionModel/blob/main/DDPM/DDPM_example.ipynb
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import multivariate_normal
import pdb


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layer):
        super().__init__()
        self.layers = nn.ModuleList( [nn.Linear(dim_in, dim_hidden)] + [nn.Linear(dim_hidden, dim_hidden) for _ in range(num_layer)] + [nn.Linear(dim_hidden, dim_out)] )
        self.gelu = nn.GELU()

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            if idx !=0 and idx != len(self.layers) - 1:
                x0 = x
                x = layer(x)
                x = x0 + self.gelu(x)
            elif idx == 0:
                x = self.gelu(layer(x))
            elif idx == len(self.layers) - 1:
                x = layer(x)
        return x


class Backbone(nn.Module):
    def __init__(self, dim_in, dim_condition, dim_hidden, num_layer, n_steps):
        super().__init__()
        self.linear_model1 = MLP(dim_in + dim_condition, dim_hidden, dim_hidden, num_layer)

        # Condition time t
        self.embedding_layer = nn.Embedding(n_steps, dim_hidden)
        self.linear_model2 = MLP(dim_hidden, dim_hidden, dim_in, num_layer)

        # condition value
        self.condition_layer = nn.Embedding(1, dim_condition)

    def forward(self, x, condition, idx):
        x_condition = torch.einsum('bi,id->bd', condition.unsqueeze(1), self.condition_layer.weight)
        x = self.linear_model2(self.linear_model1( torch.cat([x, x_condition], dim=1)) + self.embedding_layer(idx))
        return x


class Model(nn.Module):
    def __init__(self, dim_in, dim_condition, dim_hidden, num_layer, T, beta_1, beta_T):
        '''
        The epsilon predictor of diffusion process.

        beta_1    : beta_1 of diffusion process
        beta_T    : beta_T of diffusion process
        T         : Diffusion Steps
        input_dim : a dimension of data

        '''
        super().__init__()
        self.beta = torch.linspace(start = beta_1, end=beta_T, steps=T)
        self.alpha_bars = torch.cumprod(1 - self.beta, dim = 0)
        self.snr = (1 - self.alpha_bars) / self.alpha_bars

        self.backbone = Backbone(dim_in, dim_condition, dim_hidden, num_layer, T)

    def loss_fn(self, x, condition, idx=None):
        '''
        This function performed when only training phase.

        x          : real data if idx==None else perturbation data
        idx        : if None (training phase), we perturbed random index. Else (inference phase), it is recommended that you specify.

        '''
        output, epsilon, alpha_bar = self.forward(x, condition, idx=idx, get_target=True)
        loss = (output - epsilon).square().mean()
        return loss

    def forward(self, x, condition, idx=None, get_target=False):
        '''
        x          : real data if idx==None else perturbation data
        idx        : if None (training phase), we perturbed random index. Else (inference phase), it is recommended that you specify.
        get_target : if True (training phase), target and sigma is returned with output (epsilon prediction)

        '''

        if idx == None:
            idx = torch.randint(0, len(self.alpha_bars), (x.size(0), )).to(x.device)
            used_alpha_bars = self.alpha_bars[idx][:, None].to(x.device)
            epsilon = torch.randn_like(x).to(x.device)
            x_tilde = torch.sqrt(used_alpha_bars) * x + torch.sqrt(1 - used_alpha_bars) * epsilon
            
        else:
            idx = torch.Tensor([idx for _ in range(x.size(0))]).to(x.device).long()
            x_tilde = x

        output = self.backbone(x_tilde, condition, idx)        
        return (output, epsilon, used_alpha_bars) if get_target else output

    def neg_loglikelihood(self, x):
        dim_t = len(self.alpha_bars)

        idx = torch.arange(len(self.alpha_bars)).to(x.device)
        epsilon = torch.randn_like(x).to(x.device)
        epsilons = torch.einsum('t,d->td', torch.sqrt(1-self.alpha_bars).to(x.device), epsilon)
        xt = torch.einsum('t,d->td', torch.sqrt(self.alpha_bars).to(x.device), x)

        xt_q = xt + epsilons
        et = self.backbone(xt_q, idx)

        # Gaussian D_KL(m1, s1, m2, s2) = log(s2/s1) + 0.5 * (s1**2 + (m1-m2)**2) / s2**2 - 0.5

        # LT = D_KL( q(xT|x0) ; p(xT) )
        sigma_T = torch.sqrt(1 - self.alpha_bars[-1]).to(x.device)
        LT = torch.log(1 / sigma_T) + 0.5 * (sigma_T ** 2 + xt[-1].square().mean()) - 0.5

        # Lt = D_KL( q(x_t-1|xt,x0) ; p(x_t-1|xt) ) = 0.5 * ( 1 - SNR_t-1/SNRt ) ||e-et||**2
        Lt = 0.5 * ((1 - self.snr[:-1] / self.snr[1:]).to(x.device) * (et - epsilon.reshape(1,-1)).square().mean(dim=1)[1:] ).sum()

        # L0 = - log p(x0|x1) = - log( 1/sigma/sqrt(2pi) * exp( -1/2/sigma**2 * (x-mu)**2 )
        sigma_0 = torch.sqrt( (1 - self.alpha_bars[0]) / (1 - self.alpha_bars[1]) * self.beta[1] )
        L0 = - torch.log( 1 / sigma_0 / torch.sqrt(2 * torch.tensor(np.pi).to(x.device)) * torch.exp(-0.5 * (et[0] - epsilon) ** 2 ) + 1e-10 ).mean() # add 1e-10 to avoid inf

        nll = LT + Lt + L0

        return nll

