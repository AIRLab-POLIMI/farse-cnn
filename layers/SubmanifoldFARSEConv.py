import torch
from utils.rnn_utils import pack_flat_sequence, flatten_packed_sequence
from .FARSEConv import FARSEConv

class SubmanifoldFARSEConv (FARSEConv):
    def __init__(self, *args, **kwargs):
        super(SubmanifoldFARSEConv, self).__init__(*args, stride=(1, 1), **kwargs)
        self.center_id = (self.kernel_size[1] - 1)//2  + self.kernel_size[1]*((self.kernel_size[0] - 1)//2)

    def forward(self, inputs, init_state=None, grouped_events=True,  return_state=False):
        """
        :param inputs: a Tuple[Tensor,Tensor] with events and unpadded lengths if events are not already grouped, otherwise a dictionary as returned by group_events.
        :param gr_h_0: a List of Tuple[Tensor,Tensor] with the grouped initial hidden states and cell states to be passed to the lstm cells.
        :param grouped_input: True if the input is already passed as grouped by group_events, False otherwise
        :param return_state: True if the state of the module should be returned. If the state is not required, setting this to False allows to optimize execution.
        :return:
        """
        inputs = self.prepare_inputs(inputs, grouped_events, return_state=return_state)

        batch_size = inputs["batch_size"]
        rf_batch_id = inputs["batch_id"]
        rf_lengths = inputs["lengths"]
        rf_h = inputs["h"]
        rf_w = inputs["w"]
        rf_events = inputs["events"]
        rf_time = inputs["time"]
        rf_pos_id = inputs["pos_id"]

        if init_state:
            lstm_init_state = self.match_initial_states(init_state, (rf_batch_id, rf_h, rf_w))
        else:
            lstm_init_state = None

        if not (self.kernel_size[0] == 1 and self.kernel_size[1] == 1 and \
                self.stride[0] == 1 and self.stride[1] == 1):

            rf_events = self.apply_conv_weigths(rf_events, rf_pos_id)

        # summation of features inside rf occurring at same time
        rf_events, center_active, rf_time, rf_lengths = self.combine_simultaneous(rf_events, rf_pos_id, rf_time, rf_lengths)

        # converting lengths to cpu slows execution down significantly, but makes the packed sequence consistent with pack_padded_sequence
        rf_events, packing_indices = pack_flat_sequence(rf_events, rf_lengths, enforce_sorted=False)

        gr_lstm_out, gr_lstm_state = self.shared_lstm_stack(rf_events, lstm_init_state)

        gr_lstm_out = flatten_packed_sequence(gr_lstm_out, packing_indices)

        # Discard outputs when input did not have activity in the center position of the rf
        # The inputs are all processed to update the stateful representation, but only sites corresponding to the input sites produce an output
        gr_lstm_out = gr_lstm_out[center_active]
        rf_time = rf_time[center_active]
        rf_lengths = center_active.long().cumsum(dim=0)[rf_lengths.cumsum(dim=0) - 1]
        rf_lengths[1:] = rf_lengths[1:] - rf_lengths[:-1]

        if return_state:
            state = self.compose_state_dict(gr_lstm_state, (rf_batch_id, rf_h, rf_w), init_state)

            nonempty_seq_idx = rf_lengths.nonzero().squeeze(1)
            rf_lengths = rf_lengths[nonempty_seq_idx]
            rf_batch_id = rf_batch_id[nonempty_seq_idx]
            rf_h = rf_h[nonempty_seq_idx]
            rf_w = rf_w[nonempty_seq_idx]

            return {"events": gr_lstm_out, "state":state, "time": rf_time, "batch_id": rf_batch_id,
                        "lengths": rf_lengths, "h": rf_h, "w": rf_w, "batch_size": batch_size}
        else:
            return {"events": gr_lstm_out,  "time": rf_time, "batch_id": rf_batch_id,
                        "lengths": rf_lengths, "h": rf_h, "w": rf_w, "batch_size": batch_size}


    def combine_simultaneous(self, rf_events, rf_pos_id, rf_time, rf_lengths):
        max_v = rf_time.max() + 1  # add one just to avoid rare edge cases
        offsets = torch.arange(rf_lengths.shape[0], device=rf_time.device, dtype=rf_time.dtype).repeat_interleave(rf_lengths)
        offsets *= max_v
        rf_time += offsets

        new_rf_time, indices = torch.unique_consecutive(rf_time, return_inverse=True)

        new_rf_events = torch.zeros([indices.max() + 1, rf_events.shape[-1]], dtype=rf_events.dtype, device=rf_events.device)
        new_rf_events.scatter_add_(0, indices.unsqueeze(1).expand(-1, rf_events.shape[-1]), rf_events)

        is_center = (rf_pos_id == self.center_id).float()
        center_active = torch.zeros([indices.max() + 1], dtype=torch.float, device=rf_pos_id.device)
        center_active.scatter_add_(0, indices, is_center)
        center_active = center_active > 0
        # have to convert to float before scatter_add_ and then back to bool
        # scatter_add_ with bool dtype was causing undefined behaviour

        rf_groups = torch.arange(rf_lengths.shape[0], device=rf_lengths.device, dtype=rf_lengths.dtype).repeat_interleave(rf_lengths)
        new_rf_groups = torch.zeros(indices.max() + 1, device=rf_groups.device, dtype=rf_groups.dtype)
        new_rf_groups.scatter_(0, indices, rf_groups)
        _, new_rf_lengths = torch.unique_consecutive(new_rf_groups, return_counts=True)

        new_offsets = torch.arange(new_rf_lengths.shape[0], device=new_rf_time.device, dtype=new_rf_time.dtype).repeat_interleave(new_rf_lengths)
        new_offsets *= max_v
        new_rf_time -= new_offsets

        return new_rf_events, center_active, new_rf_time, new_rf_lengths


    def prepare_inputs(self, inputs, grouped_events, return_state=True):
        if not grouped_events:
            inputs = self.group_events(inputs)

        if self.kernel_size[0]==1 and self.kernel_size[1]==1:
            inputs['pos_id'] = torch.zeros(inputs['time'].shape, device=inputs['time'].device)
        else:
            # gather pixel events into receptive fields
            if return_state:
                inputs = self.gather_receptive_fields(inputs)
            else:
                inputs = self.light_gather_receptive_fields(inputs)
        return inputs


    def light_gather_receptive_fields(self, inputs):
        '''
        Optimized version of gather_receptive_fields that ignores receptive fields that will not output for the input batch, to be used for training.
        Cannot be used if the state of the layer is required, since the state of non-outputting receptive fields must also be computed.
        '''
        gr_batch_id = inputs["batch_id"]
        gr_lengths = inputs["lengths"]
        gr_h = inputs["h"]
        gr_w = inputs["w"]
        gr_events = inputs["events"]
        gr_time = inputs["time"]
        batch_size = inputs["batch_size"]

        gr_ids = (gr_batch_id * self.frame_size[0] * self.frame_size[1]) + (gr_h * self.frame_size[1]) + gr_w

        bs_id = torch.arange(batch_size, device=gr_events.device).repeat_interleave(self.rf2pixel_lut.shape[0]) \
                     .unsqueeze(1).repeat([1, self.rf2pixel_lut.shape[1]])  # shape [num_rf, num_rfcoords]

        rf2pixel_lut = self.rf2pixel_lut.repeat([batch_size, 1])
        bs_id[rf2pixel_lut == -1] = 0
        rf2pixel_lut += bs_id * self.frame_size[0] * self.frame_size[1]
        # rf2pixel_lut maps receptive fields of the whole batch to pixels (rf of different samples are independent)
        # padded pixels have index -1

        num_ids = max(gr_ids.max().item(), rf2pixel_lut.max().item()) + 2  # includes the -1 padding id

        ids_set = torch.zeros([num_ids], device=gr_ids.device, dtype=torch.bool)
        ids_set[gr_ids + 1] = True

        nonempty_ids = ids_set[rf2pixel_lut + 1]
        nonempty_ids *= nonempty_ids[:, self.center_id].unsqueeze(1)
        # set as empty all receptive fields that have no activity in their center id.
        # since we are training in a batched, synchronous scenario, we don't need to update the states of receptive fields that never output
        nonempty_rf = nonempty_ids.any(dim=1)
        num_nonempty_rf = nonempty_rf.sum()


        rf2px_flat = rf2pixel_lut.view(-1)
        rf2px_flat = rf2px_flat[nonempty_ids.view(-1)]
        # flat lookup table, filtered to contain only nonempty pixel positions

        nonempty_ids = nonempty_ids[nonempty_rf]  # discard empty rfs, shape [num_nonempty_rf, num_rfcoords]

        ids_idx = ids_set.cumsum(dim=0) - 1  # exploits the fact that gr_ids are in ascending order
        lookup_idx = ids_idx[rf2px_flat + 1]

        rf_groups = torch.arange(num_nonempty_rf, device=nonempty_rf.device)
        rf_groups = rf_groups.repeat_interleave(nonempty_ids.sum(dim=1))
        # for each consecutive sequence id in the flat lookup table, which rf it belongs to
        # (ascending number from 0 to num_nonempty_rf)

        rf_lengths = torch.zeros(num_nonempty_rf, device=gr_lengths.device, dtype=gr_lengths.dtype)
        rf_lengths.scatter_add_(0, rf_groups, gr_lengths[lookup_idx])
        # <num_nonempty_rf>, for each nonempty rf, the length of the contained sequence

        start_ids = gr_lengths.cumsum(dim=0)
        start_ids = start_ids.roll(1)
        start_ids[0] = 0

        flat_lookup_idx = start_ids[lookup_idx]  # selects the first element of every sequence in the flat tensor of sequences
        repeats = gr_lengths[lookup_idx]
        flat_lookup_idx = torch.repeat_interleave(flat_lookup_idx,
                                                  repeats)  # repeat indices for the length of each sequence
        offsets = torch.ones(flat_lookup_idx.shape[0], dtype=flat_lookup_idx.dtype, device=flat_lookup_idx.device)
        offsets[torch.cat([torch.tensor([0], device=repeats.device), repeats.cumsum(dim=0)[:-1]])] = \
            torch.cat([torch.tensor([0], device=repeats.device), (1 - repeats[:-1])])
        offsets = offsets.cumsum(dim=0)

        flat_lookup_idx += offsets  # indices that select all events of sequences in rf order

        rf_time = gr_time[flat_lookup_idx]
        rf_events = gr_events[flat_lookup_idx]
        # flat rf sequences (unsorted)

        offsets = torch.arange(rf_lengths.shape[0], device=rf_time.device, dtype=rf_time.dtype).repeat_interleave(
            rf_lengths)
        offsets *= (rf_time.max() + 1)  # add one just to avoid rare edge cases

        rf_time += offsets
        rf_time, sort_idx = torch.sort(rf_time)
        rf_time -= offsets

        rf_events = rf_events[sort_idx]
        # flat rf sequences (sorted)

        num_rfcoords = rf2pixel_lut.shape[1]
        rf_pos_id = torch.arange(num_rfcoords, device=nonempty_ids.device).repeat([num_nonempty_rf])
        rf_pos_id = rf_pos_id[nonempty_ids.view(-1)]
        rf_pos_id = torch.repeat_interleave(rf_pos_id, repeats)  # repeat pos ids for the length of each sequence
        rf_pos_id = rf_pos_id[sort_idx]

        # out coordinates are the same as in coordinates for SubmanifoldFARSEConv
        rf_w = gr_w
        rf_h = gr_h
        rf_batch_id = gr_batch_id

        return {"events": rf_events, "time": rf_time, "pos_id": rf_pos_id, "lengths": rf_lengths,
                "h": rf_h, "w": rf_w, "batch_id": rf_batch_id, "batch_size": batch_size}