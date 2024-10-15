from dataclasses import dataclass
from typing import Tuple
import numpy as np
from tonic.functional.uniform_noise import uniform_noise_numpy

@dataclass(frozen=True)
class CenterEvents:
    """Translates events to the center of the target frame size. Event coordinates should always be smaller or equal to target_size.
    Parameters:
        target_size (Tuple[int, int]): tuple of x,y target size.
    """

    target_size: Tuple[int, int]

    def __call__(self, events):
        events = events.copy()
        assert 'x' and 'y' in events.dtype.names
        assert (events['x'] < self.target_size[0]).all() and (events['y'] < self.target_size[1]).all()

        x_min = events['x'].min()
        x_max = events['x'].max()
        y_min = events['y'].min()
        y_max = events['y'].max()

        x_delta = (self.target_size[0] - 1 - x_max + x_min)//2 - x_min
        y_delta = (self.target_size[1] - 1 - y_max + y_min)//2 - y_min
        events['x'] = events['x'] + x_delta
        events['y'] = events['y'] + y_delta
        
        return events


@dataclass(frozen=True)
class RandomTranslate:
    """Translates events to a random position in the target frame. Event coordinates should always be smaller or equal to target_size.
        Parameters:
            target_size (Tuple[int, int]): tuple of x,y target size.
    """

    target_size: Tuple[int, int]

    def __call__(self, events):
        events = events.copy()
        assert 'x' and 'y' in events.dtype.names
        assert (events['x'] < self.target_size[0]).all() and (events['y'] < self.target_size[1]).all()

        x_min = events['x'].min()
        x_max = events['x'].max()
        y_min = events['y'].min()
        y_max = events['y'].max()

        x_range = self.target_size[0] - x_max + x_min
        y_range = self.target_size[1] - y_max + y_min
        x_delta = np.random.randint(0, x_range) - x_min
        y_delta = np.random.randint(0, y_range) - y_min
        events['x'] = events['x'] + x_delta
        events['y'] = events['y'] + y_delta

        return events


@dataclass(frozen=True)
class RandomCropTime:
    """ Crops a random time window of pre-defined length.
            Parameters:
                window_size (int): size of the temporal window, in seconds
    """

    crop_size: int

    def __call__(self, events):
        events = events.copy()
        assert 't' in events.dtype.names

        t_min = events['t'].min()
        t_max = events['t'].max()

        crop_size = self.crop_size * 1e6 # convert to microseconds

        crop_max_range = max(t_min, t_max - crop_size) + 1
        crop_start_t = np.random.randint(t_min, crop_max_range)

        return events[(events["t"] >= crop_start_t) & (events["t"] <= crop_start_t + crop_size)]


@dataclass(frozen=True)
class UniformNoise:
    n: int
    use_range: bool

    def __call__(self, events):
        events = events.copy()
        assert 'x' and 'y' and 't' in events.dtype.names

        x_max = events['x'].max()
        y_max = events['y'].max()

        if self.use_range:
            n = np.random.randint(0,self.n+1)
        else:
            n = self.n

        return uniform_noise_numpy(events, sensor_size=(x_max, y_max, 2), n=n)