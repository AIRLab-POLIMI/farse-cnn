import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.rnn_utils import pack_flat_sequence, flatten_packed_sequence
from .AsyncSparseModule import AsyncSparseModule


class FARSEConv (AsyncSparseModule):
    def __init__(self, input_size, hidden_size, *args, bias=True, **kwargs):
        super(FARSEConv, self).__init__(*args, **kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        if self.kernel_size[0]>1 or self.kernel_size[1]>1:
            num_rfcoords = self.kernel_size[0]*self.kernel_size[1]
            weights_size = self.input_size * self.hidden_size

            lstm_input_size =  self.hidden_size

            conv_weights = torch.nn.Parameter(torch.randn([num_rfcoords, weights_size]))
            self.register_parameter(name='conv_weights', param=conv_weights)
        else:
            lstm_input_size = self.input_size
        self.shared_lstm_stack = nn.LSTM(input_size=lstm_input_size, hidden_size=self.hidden_size, bias=bias, batch_first=False)
        self.reset_parameters()


    def reset_parameters(self):
        if self.kernel_size[0]>1 or self.kernel_size[1]>1:
            std = math.sqrt(2.0 / (self.input_size + self.hidden_size))
            with torch.no_grad():
                self.conv_weights.normal_(0, std)

        self.shared_lstm_stack.reset_parameters()


    def forward(self, inputs, init_state=None, grouped_events=True):
        """
        :param inputs: a Tuple[Tensor,Tensor] with events and unpadded lengths if events are not already grouped, otherwise a dictionary as returned by group_events.
        :param init_state: a List of Tuple[Tensor,Tensor] with the grouped initial hidden states and cell states to be passed to the lstm cells.
        :param grouped_input: True if the input is already passed as grouped by group_events, False otherwise
        :return:
        """
        inputs = self.prepare_inputs(inputs, grouped_events)

        batch_size = inputs["batch_size"]
        rf_batch_id = inputs["batch_id"]
        rf_lengths = inputs["lengths"]
        rf_h = inputs["h"]
        rf_w = inputs["w"]
        rf_events = inputs["events"]
        rf_time = inputs["time"]

        if init_state:
            lstm_init_state = self.match_initial_states(init_state, (rf_batch_id, rf_h, rf_w))
        else:
            lstm_init_state = None

        if not (self.kernel_size[0]==1 and self.kernel_size[1]==1 and\
                self.stride[0]==1 and self.stride[1]==1):
            rf_pos_id = inputs["pos_id"]
            rf_events = self.apply_conv_weigths(rf_events, rf_pos_id)

        # summation of features inside rf occurring at same time
        rf_events, rf_time, rf_lengths = self.combine_simultaneous(rf_events, rf_time, rf_lengths)

        # converting lengths to cpu slows execution down significantly, but makes the packed sequence consistent with pack_padded_sequence
        rf_events, packing_indices = pack_flat_sequence(rf_events, rf_lengths, enforce_sorted=False)

        gr_lstm_out, gr_lstm_state = self.shared_lstm_stack(rf_events, lstm_init_state)

        gr_lstm_out = flatten_packed_sequence(gr_lstm_out, packing_indices)

        state = self.compose_state_dict(gr_lstm_state, (rf_batch_id, rf_h, rf_w), init_state)

        return {"events":gr_lstm_out, "state":state, "time":rf_time, "batch_id":rf_batch_id,
                    "lengths":rf_lengths, "h":rf_h, "w":rf_w, "batch_size":batch_size}

    def apply_conv_weigths(self, rf_events, rf_pos_id):
        lstm_input_size = self.hidden_size
        new_rf_events = torch.empty([rf_events.shape[0], lstm_input_size], device=rf_events.device)
        conv_weights = self.conv_weights.view(-1, lstm_input_size, self.input_size)

        for i in range(conv_weights.shape[0]):
            new_rf_events[rf_pos_id == i] = F.linear(rf_events[rf_pos_id == i], conv_weights[i])

        return new_rf_events

    def combine_simultaneous(self, rf_events, rf_time, rf_lengths):
        max_v = rf_time.max() + 1  # add one just to avoid rare edge cases
        offsets = torch.arange(rf_lengths.shape[0], device=rf_time.device, dtype=rf_time.dtype).repeat_interleave(rf_lengths)
        offsets *= max_v
        rf_time += offsets

        new_rf_time, indices = torch.unique_consecutive(rf_time, return_inverse=True)

        new_rf_events = torch.zeros([indices.max() + 1, rf_events.shape[-1]], dtype=rf_events.dtype, device=rf_events.device)
        new_rf_events.scatter_add_(0, indices.unsqueeze(1).expand(-1, rf_events.shape[-1]), rf_events)

        rf_groups = torch.arange(rf_lengths.shape[0], device=rf_lengths.device, dtype=rf_lengths.dtype).repeat_interleave(rf_lengths)
        new_rf_groups = torch.zeros(indices.max() + 1, device=rf_groups.device, dtype=rf_groups.dtype)
        new_rf_groups.scatter_(0, indices, rf_groups)
        _, new_rf_lengths = torch.unique_consecutive(new_rf_groups, return_counts=True)

        new_offsets = torch.arange(new_rf_lengths.shape[0], device=new_rf_time.device, dtype=new_rf_time.dtype).repeat_interleave(new_rf_lengths)
        new_offsets *= max_v
        new_rf_time -= new_offsets

        return new_rf_events, new_rf_time, new_rf_lengths

    def match_initial_states(self, init_state, rf_idx):
        rf_batch_id, rf_h, rf_w = rf_idx
        rf_ids = (rf_batch_id * self.frame_size[0] * self.frame_size[1]) + (rf_h * self.frame_size[1]) + rf_w
        state_ids = (init_state["batch_id"] * self.frame_size[0] * self.frame_size[1]) + (init_state["h"] * self.frame_size[0]) + init_state["w"]

        num_ids = max(rf_ids.max().item(), state_ids.max().item()) + 1
        ids_set = torch.zeros([num_ids], device=rf_ids.device, dtype=torch.bool)
        ids_set[state_ids] = True
        ids_idx = ids_set.cumsum(dim=0) - 1
        ids_idx[~ids_set] = -1 # uninitialized rfs have idx -1
        ids_idx = ids_idx[rf_ids]

        hidden = init_state["hidden"]
        cell = init_state["cell"]

        empty_val = torch.zeros([1, 1, hidden.shape[-1]], device=hidden.device, dtype=hidden.dtype)
        hidden = torch.cat([empty_val, hidden], dim=1)
        cell = torch.cat([empty_val, cell], dim=1)

        hidden = hidden[:, ids_idx + 1]
        cell = cell[:, ids_idx + 1]

        return (hidden, cell)

    def compose_state_dict(self, lstm_state, rf_idx, prev_state=None):
        batch_id, h, w = rf_idx
        hidden = lstm_state[0]
        cell = lstm_state[1]

        if prev_state:
            state_ids = (batch_id * self.frame_size[0] * self.frame_size[1]) + (h * self.frame_size[1]) + w
            prev_ids = (prev_state["batch_id"] * self.frame_size[0] * self.frame_size[1]) + (prev_state["h"] * self.frame_size[1]) + prev_state["w"]

            num_ids = max(state_ids.max().item(), prev_ids.max().item()) + 1
            ids_set = torch.zeros([num_ids], device=state_ids.device, dtype=torch.bool)
            ids_set[state_ids] = True
            updated = ids_set[prev_ids] # for each rf in prev_state, T if it has been updated F otherwise

            prev_ids = prev_ids[~updated]
            prev_cell = prev_state["cell"][:, ~updated]
            prev_hidden = prev_state["hidden"][:, ~updated]
            # values in prev_state must be overwritten with the new updated ones

            out_ids = torch.cat([state_ids, prev_ids], dim=0)
            out_ids, sort_idx = torch.sort(out_ids)

            hidden = torch.cat([hidden, prev_hidden], dim=1)
            hidden = hidden[:, sort_idx]
            cell = torch.cat([cell, prev_cell], dim=1)
            cell = cell[:, sort_idx]

            batch_id = out_ids // (self.frame_size[0]*self.frame_size[1])
            out_ids = out_ids - (batch_id * self.frame_size[0] * self.frame_size[1])
            h = out_ids // self.frame_size[1]
            w = out_ids - (h * self.frame_size[1])

        return {"hidden":hidden, "cell":cell, "h":h, "w":w, "batch_id":batch_id}


    def compute_flops(self, inputs, grouped_events=True):
        x = self.prepare_inputs(inputs, grouped_events)

        rf_lengths = x["lengths"]
        rf_events = x["events"]
        rf_time = x["time"]

        n_ev = rf_events.shape[0]
        input_size = rf_events.shape[-1]
        out_size = self.hidden_size

        # matrix-vector mult for all inputs with input matrix
        flops = n_ev * 4 * out_size * (2 * input_size - 1)

        # aggregation of simultaneous inputs at same cells
        max_v = rf_time.max() + 1
        offsets = torch.arange(rf_lengths.shape[0], device=rf_time.device, dtype=rf_time.dtype).repeat_interleave(rf_lengths)
        offsets *= max_v
        rf_time = rf_time + offsets
        agg_rf_time, count = torch.unique_consecutive(rf_time, return_counts=True)
        agg_n = rf_time.shape[0] - agg_rf_time.shape[0]  # number of inputs that have been aggregated
        flops = flops + agg_n * (4 * out_size)

        # cell update
        n_ev = agg_rf_time.shape[0]  # number of events after aggregation
        flops = flops + n_ev * (4 * out_size * (2 * out_size - 1) + 8 * out_size + 4 * out_size)

        return flops