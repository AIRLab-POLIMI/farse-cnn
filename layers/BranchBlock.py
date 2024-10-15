import torch
import torch.nn as nn
from .AsyncSparseModule import AsyncSparseModule
from .SubmanifoldFARSEConv import SubmanifoldFARSEConv
from .FARSEConv import FARSEConv

class BranchBlock(nn.Module):
    def __init__(self, frame_size, input_size, branch_1=None, branch_2=None, merge_func='add'):
        super(BranchBlock, self).__init__()

        self.merge_func = merge_func
        if self.merge_func not in ['add','cat']:
            raise ValueError('merge_func = %s is not valid.' % self.merge_func)

        frame_size1, hidden_size1 = frame_size, input_size
        if branch_1:
            self.branch_1 = branch_1
            frame_size1 = branch_1[-1].frame_output_size
            for l in self.branch_1:
                hidden_size1 = getattr(l, "hidden_size", hidden_size1)
        else:
            self.branch_1 = nn.ModuleList([nn.Identity()])

        frame_size2, hidden_size2 = frame_size, input_size
        if branch_2:
            self.branch_2 = branch_2
            frame_size2 = branch_2[-1].frame_output_size
            for l in self.branch_2:
                hidden_size2 = getattr(l, "hidden_size", hidden_size2)
        else:
            self.branch_2 = nn.ModuleList([nn.Identity()])


        if frame_size1 == frame_size2:
            self.frame_output_size = frame_size1
        else:
            raise Exception("Output frame size of branches is not the same. branch 1 = %s, branch 2 = %s" % (str(frame_size1), str(frame_size2)))

        if self.merge_func == 'add':
            self.hidden_size = max(hidden_size1, hidden_size2)
            if self.hidden_size > hidden_size1:
                self.proj_layer = nn.Linear(hidden_size1, self.hidden_size)
            elif self.hidden_size > hidden_size2:
                self.proj_layer = nn.Linear(hidden_size2, self.hidden_size)

        elif self.merge_func == 'cat':
            self.hidden_size = hidden_size1 + hidden_size2


    def forward(self, x, init_state=(None,None), return_state=False):
        x1,states1 = self.exec_branch(self.branch_1, x, init_state[0], return_state)
        x2,states2 = self.exec_branch(self.branch_2, x, init_state[1], return_state)

        if (x1['lengths']!=x2['lengths']).any():
            raise Exception('Event streams of branches have different lengths. %s != %s' % (str(x1['lengths']),str(x2['lengths'])))
        if (x1['time'] != x2['time']).any():
            raise Exception('Event streams of branches have different timestamps.')
        if (x1['h'] != x2['h']).any() or (x1['w'] != x2['w']).any():
            raise Exception('Event streams of branches have different spatial coordinates.')

        if self.merge_func == 'add':
            if x1['events'].shape[-1] < self.hidden_size:
                x1['events'] = self.proj_layer(x1['events'])
            elif x2['events'].shape[-1] < self.hidden_size:
                x2['events'] = self.proj_layer(x2['events'])
            events = x1['events'] + x2['events']

        elif self.merge_func == 'cat':
            events = torch.cat([x1['events'],x2['events']], dim=-1)

        out_dict = {"events": events, "time": x1['time'], "lengths": x1['lengths'],
                    "batch_id": x1['batch_id'], "h": x1['h'], "w": x1['w'], "batch_size": x1['batch_size']}
        if return_state:
            out_dict['state'] = (states1, states2)

        return out_dict

    def exec_branch(self, branch, x, init_state=None, return_state=False):
        if not init_state:
            init_state = [None] * len(branch)
        states = []
        for l,s in zip(branch,init_state):
            if isinstance(l, SubmanifoldFARSEConv):
                x = l(x, init_state=s, return_state=return_state)
            elif isinstance(l, FARSEConv):
                x = l(x, init_state=s)
            else:
                x = l(x)
            if return_state:
                states.append(x.pop('state', None))

        return x, states


    def compute_flops(self, x):
        flops = 0

        x1 = x
        for l in self.branch_1:
            if isinstance(l, AsyncSparseModule):
                flops = flops + l.compute_flops(x1)
            x1 = l(x1)

        x2 = x
        for l in self.branch_2:
            if isinstance(l, AsyncSparseModule):
                flops = flops + l.compute_flops(x2)
            x2 = l(x2)

        if self.merge_func == 'add':
            if x1['events'].shape[-1] < self.hidden_size:
                flops = flops + 2*x1['events'].shape[-1]*self.hidden_size
            elif x2['events'].shape[-1] < self.hidden_size:
                flops = flops + 2*x2['events'].shape[-1]*self.hidden_size
            flops = flops + self.hidden_size

        elif self.merge_func == 'cat':
            pass # no FLOPS performed for concat

        return flops