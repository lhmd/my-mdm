import torch

# lengths_to_mask(lengths, max_len) 函数：

# 目的：根据序列长度生成掩码矩阵，掩码矩阵用于指示每个序列中的有效数据位置。
# 参数：lengths 是一个包含各序列实际长度的列表或张量，max_len 是序列的最大长度。
# 功能：该函数创建一个二维张量（矩阵），其中的元素通过比较序列最大长度与实际长度生成布尔值，即矩阵中的每个元素代表序列中相应位置是有效数据（True）还是填充数据（False）。
# collate_tensors(batch) 函数：

# 目的：将一个批次中的多个张量整合到一个大的张量中，常用于处理具有不同形状的数据。
# 参数：batch 是一个包含多个张量的列表，这些张量代表不同数据样本。
# 功能：该函数首先确定批次中所有张量的最大尺寸，然后创建一个足够大的"画布"张量，并将每个输入张量复制到这个"画布"的相应位置上。
# collate(batch) 函数：

# 目的：整合一个批次的数据，包括输入、长度、掩码等，并处理可能存在的不同类型的数据（如文本或动作数据）。
# 参数：batch 是包含多个数据样本的列表，每个样本可以包含不同的数据类型和结构。
# 功能：该函数处理批次中的数据，创建数据张量、长度张量和掩码张量，并根据需要整合额外的信息（如文本或动作数据）。最终，它返回处理后的数据和一个包含附加条件信息的字典。
# t2m_collate(batch) 函数：

# 目的：是collate函数的一个适配器，用于处理特定结构的数据批次。
# 参数：batch 是包含多个数据样本的列表，每个样本具有特定的结构。
# 功能：该函数适配输入数据的结构，以满足collate函数的要求，然后调用collate函数进行处理。

def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]


    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})

    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({'tokens': textbatch})

    if 'action' in notnone_batches[0]:
        actionbatch = [b['action'] for b in notnone_batches]
        cond['y'].update({'action': torch.as_tensor(actionbatch).unsqueeze(1)})

    # collate action textual names
    if 'action_text' in notnone_batches[0]:
        action_text = [b['action_text']for b in notnone_batches]
        cond['y'].update({'action_text': action_text})

    return motion, cond

# an adapter to our collate func
def t2m_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[2], #b[0]['caption']
        'tokens': b[6],
        'lengths': b[5],
    } for b in batch]
    return collate(adapted_batch)
