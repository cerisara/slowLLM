import torch

alibi = torch.ones((112,1,34),dtype=torch.bfloat16)
v1 = torch.ones((112,34,128),dtype=torch.bfloat16)
v2 = torch.ones((112,128,34),dtype=torch.bfloat16)

alibi.cuda()
v1.cuda()
v2.cuda()

matmul_result = alibi.baddbmm(
    batch1=v1,
    batch2=v2,
    beta=1.0,
    alpha=0.08,
)


