from typing import Any, Iterable, Iterator, List, Optional, Union, Sequence, Tuple, cast

import torch
from torch import Tensor, nn
import torch.autograd
import torch.cuda
from .worker import Task, create_workers
from .partition import _split_module

# ASSIGNMENT 4.2
def _clock_cycles(num_batches: int, num_partitions: int) -> Iterable[List[Tuple[int, int]]]:
    '''Generate schedules for each clock cycle.

    An example of the generated schedule for m=3 and n=3 is as follows:
    
    k (i,j) (i,j) (i,j)
    - ----- ----- -----
    0 (0,0)
    1 (1,0) (0,1)
    2 (2,0) (1,1) (0,2)
    3       (2,1) (1,2)
    4             (2,2)

    where k is the clock number, i is the index of micro-batch, and j is the index of partition.

    Each schedule is a list of tuples. Each tuple contains the index of micro-batch and the index of partition.
    This function should yield schedules for each clock cycle.
    '''
    # BEGIN SOLUTION
    return [[(batch, sum-batch) for batch in range(num_batches) if sum-batch>=0 and sum-batch<num_partitions] for sum in range(num_batches+num_partitions-1)]
    # END SOLUTION

class Pipe(nn.Module):
    def __init__(
        self,
        module: nn.ModuleList,
        split_size: int = 1,
    ) -> None:
        super().__init__()

        self.split_size = int(split_size)
        self.partitions, self.devices = _split_module(module)
        (self.in_queues, self.out_queues) = create_workers(self.devices)

    # ASSIGNMENT 4.2
    def forward(self, x):
        ''' Forward the input x through the pipeline. The return value should be put in the last device.

        Hint:
        1. Divide the input mini-batch into micro-batches.
        2. Generate the clock schedule.
        3. Call self.compute to compute the micro-batches in parallel.
        4. Concatenate the micro-batches to form the mini-batch and return it.
        '''
        # BEGIN SOLUTION
        mini_batch_size = x.shape[0]
        # avoid mini_batch_size < split_size
        self.split_size = min(self.split_size, mini_batch_size)
        micro_batch_size = mini_batch_size // self.split_size
        micro_batches = [x[i:i+micro_batch_size] for i in range(0, mini_batch_size, micro_batch_size)]
        num_partitions = len(self.partitions)
        schedules = _clock_cycles(self.split_size, num_partitions)
        ''' _clock_cycles(8, 2)
                (batch, device)
                [[(0, 0)],
                 [(0, 1), (1, 0)], 
                 [(1, 1), (2, 0)], 
                 [(2, 1), (3, 0)], 
                 [(3, 1), (4, 0)], 
                 [(4, 1), (5, 0)], 
                 [(5, 1), (6, 0)], 
                 [(6, 1), (7, 0)], 
                 [(7, 1)]]
        '''
        for schedule in schedules:
            self.compute(micro_batches, schedule)
        return torch.concat(micro_batches)
        # END SOLUTION

    # ASSIGNMENT 4.2
    def compute(self, batches, schedule: List[Tuple[int, int]]) -> None:
        '''Compute the micro-batches in parallel.

        Hint:
        1. Retrieve the partition and microbatch from the schedule.
        2. Use Task to send the computation to a worker. 
        3. Use the in_queues and out_queues to send and receive tasks.
        4. Store the result back to the batches.
        '''
        partitions = self.partitions
        devices = self.devices
        # BEGIN SOLUTION
        for microbatch, partition in schedule:
            batches[microbatch] = batches[microbatch].to(devices[partition])
            self.in_queues[partition].put(Task(partitions[partition], batches[microbatch]))
        for microbatch, partition in schedule:
            batches[microbatch] = self.out_queues[partition].get()[1][1]
        # END SOLUTION

