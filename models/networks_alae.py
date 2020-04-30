import torch.nn as nn
from torch.nn import functional as F

class MappingBlock(nn.Module):
    def __init__(self, inputs, output, lrmul):
        super(MappingBlock, self).__init__()
        self.fc = ln.Linear(inputs, output, lrmul=lrmul)

    def forward(self, x):
        x = F.leaky_relu(self.fc(x), 0.2)
        return x

class VAEMappingFromLatent(nn.Module):
    def __init__(self, num_layers, mapping_layers=5, latent_size=256, mapping_fmaps=256):
        super(VAEMappingFromLatent, self).__init__()
        input = latent_size
        for i in range(mapping_layers):
            outputs = latent_size if i == mapping_layers - 1 else mapping_fmaps
            block  =




    def forward(self, x):
        return x