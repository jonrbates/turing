import torch
import torch.nn.functional as F

def hard_max_attention(query, key, value, w_q, b_q, w_k, b_k, w_v, b_v, k_0, v_0):
    """A hard max attention transformer layer    

    Args:
        query : (L, E)
        key: (S, E)
        value: (S, E)
        w_q, b_q : input projection weight and bias for query
        w_k, b_k : input projection weight and bias for key
        w_v, b_v : input projection weight and bias for value
        k_0, v_0 : null key and null value

    Notes:
        See functional.multi_head_attention_forward() for softmax

    """
    L, S = query.size(0), key.size(0)
    assert query.ndim == 2
    assert key.ndim == 2
    assert value.ndim == 2
    assert k_0.ndim == 1
    assert v_0.ndim == 1
    query = query.unsqueeze(1)
    key = key.unsqueeze(1)
    value = value.unsqueeze(1)
    q, k, v = F.linear(query, w_q, b_q), F.linear(key, w_k, b_k), F.linear(value, w_v, b_v)
    q = q.squeeze(dim=1)
    k = k.squeeze(dim=1)
    v = v.squeeze(dim=1)
    k_0 = k_0.unsqueeze(0)
    v_0 = v_0.unsqueeze(0)
    k = torch.cat([k_0, k])
    v = torch.cat([v_0, v])
    tau = torch.mm(q, k.transpose(0, 1))
    # TODO: remove iteration
    max_values = torch.max(tau, 1).values
    transfer = torch.zeros(L, S+1)
    for i, val in enumerate(max_values):
        idx = torch.isclose(tau[i, :], val)
        transfer[i, idx] = 1
    transfer = F.normalize(transfer, p=1, dim=1)
    output = torch.mm(transfer, v)
    return output