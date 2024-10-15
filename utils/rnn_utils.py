from typing import List, Tuple
import torch
import torch.nn.utils.rnn as rnn

"""
Packs a flat tensor of consecutive sequences into a PackedSequence, given the lengths of each sequence in the input tensor.
Returns the PackedSequence and the packing_indices that are required to perform the unpacking operation.
"""
def pack_flat_sequence(input: torch.Tensor, lengths: torch.Tensor, enforce_sorted=True):
    if enforce_sorted:
        sorted_indices = torch.arange(lengths.shape[0], device=lengths.device, dtype=torch.int64)
    else:
        sorted_indices = torch.argsort(lengths, descending=True)

    ranges = torch.arange(lengths.max(), device=lengths.device, dtype=lengths.dtype)
    ranges = ranges.expand(lengths.shape[0], -1)
    batch_sizes = (ranges < lengths.unsqueeze(1)).sum(dim=0)

    batch_summed = batch_sizes.cumsum(dim=0)
    batch_summed = batch_summed.roll(1)
    batch_summed[0] = 0
    idx = torch.arange(input.shape[0], device=batch_sizes.device, dtype=batch_sizes.dtype)
    idx -= torch.repeat_interleave(batch_summed, batch_sizes)

    offsets = torch.arange(batch_sizes.shape[0], device=batch_sizes.device, dtype=batch_sizes.dtype)
    offsets = torch.repeat_interleave(offsets, batch_sizes)

    start_ids = lengths.cumsum(dim=0)
    start_ids = start_ids.roll(1)
    start_ids[0] = 0

    packing_indices = start_ids[sorted_indices[idx]] + offsets
    packing_indices = packing_indices.to(input.device)
    data = input[packing_indices]

    # as required by the PackedSequence doc, batch_sizes should always reside on cpu, sorted_indices should always reside on the same device as data
    return rnn.PackedSequence(data, batch_sizes.cpu(), sorted_indices.to(data.device), None), packing_indices

"""
Inverse operation of pack_flat_sequence. Takes as input the PackedSequence and the packing indices, and returns a flat tensor of consecutive sequences.
"""
def flatten_packed_sequence(packed_sequence: rnn.PackedSequence, packing_indices: torch.Tensor):
    data = packed_sequence.data
    flat_sequence = torch.zeros_like(data)
    flat_sequence.scatter_(0, packing_indices.unsqueeze(1).expand(-1, data.shape[1]), data)

    return flat_sequence