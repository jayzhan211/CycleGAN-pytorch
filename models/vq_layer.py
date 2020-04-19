import torch.nn as nn
import torch
from torch.nn import functional as F


class VectorQuantizerEMA(nn.Module):
    def __init__(self, dim, n_embed, commitment_cost, decay=0.99, eps=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

        self._w = torch.randn(dim, n_embed)
        self.embed_cluster_size = torch.zeros(n_embed)
        self.embed_avg = torch.clone(self._w)

    def forward(self, input):
        flatten = input.view(-1, self.dim)
        dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self._w
                + self._w.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        # embed_onehot.size = (hw,n_embed)
        embed_ind = embed_ind.view(*input.size()[:-1])

        quantize = self.embed_code(embed_ind)
        e_latent_loss = (quantize.detach() - input).pow(2).mean()

        if self.training:
            self.embed_cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, embed_onehot.sum(0)
            )
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum)
            n = self.embed_cluster_size.sum()
            cluster_size = (
                    (self.embed_cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self._w.data.copy_(embed_normalized)

        loss = self.commitment_cost * e_latent_loss
        # quantize = input + (quantize - input).detach()
        avg_probs = torch.mean(embed_onehot, 0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, perplexity

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self._w.transpose(0, 1))
