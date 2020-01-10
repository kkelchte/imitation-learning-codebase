
import numpy as np

from src.data.data_types import Frame
from src.sim.common.data_types import SensorType


def convert_state_to_frame(raw_data: np.array, sensor_type: SensorType, time_stamp_us: int) -> Frame:
    """Converts numpy array into savable frame format.
    :param raw_data: raw data as numpy array
    :param sensor_type: string indicating saving format
    :param time_stamp_us: integer indicating the frames milliseconds from start of episode
    :return:
    """
    return Frame(
        sensor_type=sensor_type,
        time_stamp_us=time_stamp_us,
        data=raw_data
    )


def timestamp_to_filename(time_stamp_us: int) -> str:
    """Convert timestamp from milliseconds to string value
    :param time_stamp_us: [milliseconds]
    :return:
    """
    return f'{time_stamp_us:010d}'
