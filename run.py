
''' Simple script file to execute AlbertModel

- Example
```
python run.py
```

'''
import numpy as np
import torch

from modeling.albert_modeling import AlbertModel
from config.albert_config import AlbertConfig
from utils import get_num_params

# set config
conf = AlbertConfig()
np.random.seed(100)
n_batch = 3

# make random input
inp = np.random.randint(0, conf.n_vocab, (n_batch, 10))
inp = torch.LongTensor(inp)
print(inp.shape, inp.sum())

# define AlbertModel
transformer = AlbertModel(conf)

# output AlbertModel
out, pout = transformer(inp)
print(out.shape, pout.shape)
print(out.sum(), pout.sum())
