import copy
import os
from copy import deepcopy
from typing import List, Tuple, Union

import PIL
from cv2 import cv2
import h5py
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.core.data_types import Dataset
from src.core.data_types import Experience

#############################################################
#  Helper functions to store data used by data_saver.py     #
#############################################################
from src.core.utils import get_data_dir


def timestamp_to_filename(time_stamp_ms: int) -> str:
    return f'{time_stamp_ms:015d}'


def store_image(data: np.ndarray, file_name: str) -> None:
    if data.dtype in [np.float32, np.float16, np.float64]:
        if not (np.amin(data) >= 0 and np.amax(data) <= 1):
            data += np.amin(data)
            data /= np.amax(data)
        data = (data * 255).astype(np.uint8)
    else:
        assert data.dtype == np.uint8
    if np.argmin(data.shape) == 0 and len(data.shape) == 3:  # got a channel first image
        data = data.swapaxes(0, 1).swapaxes(1, 2)
    im = Image.fromarray(data)
    im.save(file_name)  # await


def store_array_as_numpy(data: np.ndarray, file_name: str) -> None:
    np.save(file_name, data)


def store_array_to_file(data: np.ndarray, file_name: str, time_stamp: int = 0) -> None:
    value = ' '.join(f'{x:0.5f}' for x in data) if len(data.shape) > 0 else str(data)
    message = f'{time_stamp} : ' + value + '\n'
    with open(file_name, 'a') as f:
        f.write(message)

##############################################################################
#  Helper functions data loading and preprocessing used by data_loader.py #
##############################################################################


def filename_to_timestamp(filename: str) -> Union[None, int]:
    try:
        result = int(os.path.basename(filename).split('.')[0])
    except ValueError:
        result = None
    return result


def load_and_preprocess_file(file_name: str, size: tuple = None, scope: str = 'default') -> Union[None, torch.Tensor]:
    """
    file_name: full path to image file
    size: tuple or list of 3 dimensions: (CHANNEL, WIDTH, HEIGHT) used with
    range: 'default' ~ 0:1, 'uint8' ~ 0:255, 'zero_centered' ~ -1:1
    """
    try:
        data = Image.open(file_name, mode='r')
    except PIL.UnidentifiedImageError:
        return None
    if size is not None and len(size) != 0:
        data = cv2.resize(np.asarray(data), dsize=(size[1], size[2]), interpolation=cv2.INTER_LANCZOS4)
        if size[0] == 1:
            data = data.mean(axis=-1, keepdims=True)
    # Image PIL loads data in uint8 ==> divide by 255 to get float 0:1
    data = np.array(data).astype(np.float32)/255.  # uint8 -> float32

    if np.amin(data) < 0:
        data -= np.amin(data)
    if np.amax(data) > 1:
        data /= np.amax(data)

    if scope == 'zero_centered':
        data *= 2
        data -= 1

    # check correct range
    if scope == "default":
        assert np.amax(data) <= 1 and np.amin(data) >= 0
    elif scope == "zero_centered":
        assert 0 <= np.amax(data) <= 1 and -1 <= np.amin(data) < 0

    data = data.swapaxes(1, 2).swapaxes(0, 1)  # make channel first
    return torch.as_tensor(data, dtype=torch.float32)


def load_data_from_directory(directory: str, size: tuple = None, scope: str = 'default') -> Tuple[list, list]:
    """
    directory: full path containing the observations
    size: tuple or list of 3 dimensions: (CHANNEL, WIDTH, HEIGHT)
    """
    time_stamps = []
    data = []
    for f in sorted(os.listdir(directory)):
        data.append(load_and_preprocess_file(file_name=os.path.join(directory, f),
                                             size=size,
                                             scope=scope))
        time_stamps.append(filename_to_timestamp(f))
    return time_stamps, data


def load_data_from_file(filename: str, size: tuple = ()) -> Tuple[list, list]:
    """
    filename: full path to data file
    size: tuple or list of desired sample dimensions used with a simple numpy reshape.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    time_stamps = []
    data = []
    for line in lines:
        time_stamp, data_stamp = line.strip().split(':')
        time_stamps.append(float(time_stamp))
        data_vector = torch.as_tensor([float(d) for d in data_stamp.strip().split(' ')], dtype=torch.float32)
        if size is not None and len(size) != 0:
            data_vector = data_vector.reshape(size)
        data.append(data_vector)
    return time_stamps, data


def load_data(dataype: str, directory: str, size: tuple = None, scope: str = 'default') -> Tuple[list, list]:
    if os.path.isdir(os.path.join(directory, dataype)):
        return load_data_from_directory(os.path.join(directory, dataype), size=size, scope=scope)
    elif os.path.isfile(os.path.join(directory, dataype)):
        return load_data_from_file(os.path.join(directory, dataype), size=size)
    else:
        return [], []


def arrange_run_according_timestamps(run: dict, time_stamps: dict) -> dict:
    """Ensure there is a data row in the torch tensor for each time stamp.
    """
    clean_run = {
        k: [] for k in run.keys()
    }
    while min([len(time_stamps[k]) for k in time_stamps.keys()]) != 0:
        # get first coming time stamp
        current_time_stamp = min([time_stamps[data_type][0] for data_type in time_stamps.keys()])
        # check if all field in run have a value for this stamp
        check = True
        for k in time_stamps.keys():
            check = time_stamps[k][0] == current_time_stamp and check
        if check:  # if check, add tensor to current tensors
            for k in run.keys():
                clean_run[k].append(run[k][0])
        # discard data corresponding to this timestamp
        for k in run.keys():
            while len(time_stamps[k]) != 0 and time_stamps[k][0] == current_time_stamp:
                run[k] = run[k][1:] if len(run[k]) > 1 else []
                time_stamps[k] = time_stamps[k][1:] if len(time_stamps[k]) > 1 else []
    return clean_run


def torch_append(destination: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    #  Unused
    if destination.size() != (0,):
        assert destination[0].size() == source[0].size()
    return torch.cat((destination, source), 0)


def load_run(directory: str, arrange_according_to_timestamp: bool = False, input_size: List[int] = None,
             scope: str = 'default') -> List[Experience]:
    run = {}
    time_stamps = {}
    for x in os.listdir(directory):
        try:
            k = x if not x.endswith('.data') else x[:-5]
            time_stamps[x], run[k] = load_data(x, directory, size=input_size if k == 'observation' else None,
                                               scope=scope if k == 'observation' else None)
        except:
            pass
    if arrange_according_to_timestamp:
        run = arrange_run_according_timestamps(run, time_stamps)
    if len(run.keys()) == 0:
        return []
    else:
        return [Experience(
            observation=run['observation'][index] if 'observation' in run.keys() else None,
            action=run['action'][index] if 'action' in run.keys() else None,
            reward=run['reward'][index] if 'reward' in run.keys() else None,
            done=run['done'][index] if 'done' in run.keys() else None
        ) for index in range(len(run['observation']))]

####################################################################
#  Helper functions to upweight data sampling in data_loader.py    #
####################################################################


def calculate_weights(data: List[float], number_of_bins: int = 3) -> List[float]:
    number_of_bins = min(len(set(data)), number_of_bins)
    max_steps = len(data)
    x_min = min(data)
    x_max = max(data)
    x_width = x_max - x_min
    if x_width == 0:
        return [1.] * len(data)
    # start with uniform bins
    bin_borders = [x_min + index * float(x_width) / number_of_bins for index in range(number_of_bins)]
    bin_borders += [x_max + 10e-5]
    assert len(bin_borders) == number_of_bins + 1
    for step in range(max_steps):
        # count samples in each bin
        num_samples_in_bins = [sum(np.digitize(data, bin_borders) == index + 1) for index in range(number_of_bins)]
        if max(num_samples_in_bins) - min(num_samples_in_bins) < np.ceil(len(data) * 5. / 100):
            break
        # shift border towards lower number of samples
        fresh_borders = [x_min]
        delta = 10e-4 * x_width
        for index in range(number_of_bins - 1):
            fresh_borders += [
                min(bin_borders[index + 2] - 10e-5,  # don't pass higher bound
                    max(fresh_borders[index] + 10e-5,  # don't pass lower bound
                        bin_borders[index + 1] +  # original border
                        np.sign(num_samples_in_bins[index + 1] - num_samples_in_bins[index]) * delta)  # gradientdescent
                    )]
        fresh_borders += [x_max + 10e-5]
        assert min(np.diff(fresh_borders)) >= 0, f"Non-monotonic bin_borders: {fresh_borders}"
        bin_borders = deepcopy(fresh_borders)
    else:
        print(f'Failed to converge weight balancing {max_steps} steps with data [{x_min}: {x_max}] '
              f'and {number_of_bins} bins: {bin_borders}')
    num_samples_in_bins = [sum(np.digitize(data, bin_borders) == index + 1) for index in range(number_of_bins)]
    weights_for_each_bin = [
        (bin_borders[index+1]-bin_borders[index])/x_width * len(data)/num_samples_in_bins[index]
        for index in range(number_of_bins)
    ]
    return [weights_for_each_bin[d - 1] for d in np.digitize(data, bin_borders)]


def balance_weights_over_actions(dataset: Dataset) -> List[float]:
    actions = dataset.actions
    action_dimension = actions[0].squeeze().size()
    assert len(action_dimension) == 0 or len(action_dimension) == 1
    if len(action_dimension) == 0:
        weights = calculate_weights(data=[float(a) for a in actions])
        return [float(w)/sum(weights) for w in weights]
    else:
        weights = {}
        for dim in range(action_dimension[0]):
            weights[dim] = calculate_weights(data=[float(a[dim]) for a in actions])
        mean_over_dimensions = [
            np.mean([weights[dim][index] for dim in range(action_dimension[0])]) for index in range(len(actions))
        ]
        return [float(w)/sum(mean_over_dimensions) for w in mean_over_dimensions]


def get_ideal_number_of_bins(data: List[float]) -> int:
    import warnings
    warnings.warn('DEPRECATED')
    number_of_bins = 1
    heights, boundaries, _ = plt.hist(data, bins=number_of_bins)
    while min(heights) > 0.05 * len(data):  # minimal bin should ideally contain 5% of the data
        number_of_bins += 1
        heights, boundaries, _ = plt.hist(data, bins=number_of_bins)
    number_of_bins -= 1
    return number_of_bins


def calculate_probabilities(data: List[float]) -> List[float]:
    import warnings
    warnings.warn('DEPRECATED')
    number_of_bins = get_ideal_number_of_bins(data)
    if number_of_bins == 0:
        return [1./len(data)]*len(data)
    heights, boundaries, _ = plt.hist(data, bins=number_of_bins)

    normalized_inverse_heights = heights ** -1 / sum(heights ** -1)
    probabilities = []
    for d in data:
        # loop over boundaries to detect correct bin index
        index = 0
        for i, b in enumerate(boundaries[:-1]):
            index = i
            if b >= d:
                break
        probabilities.append(normalized_inverse_heights[index])
    # normalize probabilities
    return [p/sum(probabilities) for p in probabilities]

###################################################################
#  Helper functions to create or load HDF5 file                   #
###################################################################


def create_hdf5_file_from_dataset(filename: str, dataset: Dataset) -> None:
    h5py_file = h5py.File(filename, 'w')
    h5py_dataset = h5py_file.create_group('dataset')
    for tag, data in zip(['observations', 'actions', 'rewards', 'done'],
                         [dataset.observations, dataset.actions, dataset.rewards, dataset.done]):
        if None not in data:
            h5py_dataset[tag] = np.asarray([o.numpy() for o in data])
    h5py_file.close()


def load_dataset_from_hdf5(filename: str, input_size: List[int] = None) -> h5py.Group:
    h5py_file = h5py.File(filename, 'r')
    if input_size is not None and input_size != []:
        assert tuple(h5py_file['dataset']['observations'][0].shape) == tuple(input_size)
    print(len(h5py_file['dataset']['done'][:]))
    return h5py_file['dataset']


def add_run_to_h5py(h5py_file: h5py.File, run: str) -> h5py.File:
    import warnings
    warnings.warn('DEPRECATED')
    data = {}
    time_stamps = {}
    # load all data from run
    for element_name in [element for element in os.listdir(run) if os.path.isdir(os.path.join(run, element))
                                                                   or element.endswith('.data')]:
        element_path = os.path.join(run, element_name)
        time_stamps[element_name], data[element_name] = load_data_from_file(element_path) \
            if element_name.endswith('.data') else load_data_from_directory(element_path)
    if len(data.keys()) == 0:
        return h5py_file
    # arrange all data to have corresponding time stamps
    clean_data = {
        element_name: torch.Tensor() for element_name in data.keys()
    }
    while min([len(time_stamps[element_name]) for element_name in time_stamps.keys()]) != 0:
        # get first coming time stamp
        current_time_stamp = min([time_stamps[element_name][0] for element_name in time_stamps.keys()])
        # check if all elements have a value for this stamp
        check = True
        for element_name in data.keys():
            check = time_stamps[element_name][0] == current_time_stamp and check
        if check:
            for element_name in data.keys():
                clean_data[element_name] = torch_append(clean_data[element_name], data[element_name][0].unsqueeze_(0))
        for element_name in data.keys():
            while len(time_stamps[element_name]) != 0 and time_stamps[element_name][0] == current_time_stamp:
                data[element_name] = data[element_name][1:] if len(data[element_name]) > 1 else []
                time_stamps[element_name] = time_stamps[element_name][1:] if len(time_stamps[element_name]) > 1 else []
    del data
    # final clean up check
    lengths = [len(value) for value in clean_data.values()]
    assert min(lengths) == max(lengths)
    if lengths != 0:
        run_group = h5py_file.create_group(os.path.basename(run))
        # each data element becomes a separate dataset in the h5py_file
        for element_name in clean_data.keys():
            run_group[element_name] = clean_data[element_name].numpy()
    return h5py_file


def create_hdf5_file_from_run_directories(filename: str, runs: List[str]) -> None:
    import warnings
    warnings.warn('DEPRECATED')
    h5py_file = h5py.File(filename, 'w')
    for run in tqdm(runs, ascii=True, desc=f'creating {os.path.basename(filename)}'):
        h5py_file = add_run_to_h5py(h5py_file=h5py_file, run=run)


def select(data: Union[list, torch.Tensor, np.ndarray, Dataset], indices: List[int]) -> Union[list, torch.Tensor,
                                                                                              np.ndarray, Dataset]:
    if isinstance(data, list):
        return [data[i] for i in indices]
    elif isinstance(data, Dataset):
        return Dataset(
            observations=[data.observations[i] for i in indices],
            actions=[data.actions[i] for i in indices],
            rewards=[data.rewards[i] for i in indices],
            done=[data.done[i] for i in indices],
        )
    else:  # assuming Tensor or numpy array
        return data.squeeze()[indices]

###################################################################
#  Helper functions to create or load HDF5 file                   #
###################################################################


def parse_binary_maps(observations: List[torch.Tensor], invert: bool = False) -> List[np.ndarray]:
    '''
    create binary maps from observations based on R-channel which is 1 on background and 0 at blue line.
    :param observations: [channels (RGB) x height x width ]
    :param invert: bool switch high - low binary values
    :return: binary maps : [1 x height x width ]
    '''
    observations = [o.numpy() for o in observations]
    binary_images = [
        cv2.threshold(o.mean(axis=0), 0.5, 1, cv2.THRESH_BINARY if not invert else cv2.THRESH_BINARY_INV)[1]
        for o in observations
    ]
    return binary_images


def set_binary_maps_as_target(dataset: Dataset, invert: bool = False, binary_images: List[np.ndarray] = None) -> Dataset:
    if binary_images is None:
        binary_images = parse_binary_maps(copy.deepcopy(dataset.observations), invert=invert)
    dataset.actions = [torch.as_tensor(b) for b in binary_images]
    return dataset


def augment_background_noise(dataset: Dataset, p: float = 0.0, low: float = 0., high: float = 1.,
                             binary_images: List[np.ndarray] = None) -> Dataset:
    """
    parse background and fore ground. Give fore ground random color and background noise.
    :param binary_images:
    :param high:
    :param low:
    :param p: probability for each image to be augmented with background
    :param dataset: augmented dataset in which observations are adjusted with new back- and foregrounds
    :return: augmented dataset
    """
    if binary_images is None:
        binary_images = parse_binary_maps(copy.deepcopy(dataset.observations), invert=True)
    augmented_dataset = copy.deepcopy(dataset)
    augmented_dataset.observations = []
    num_channels = dataset.observations[0].shape[0]
    for index, image in tqdm(enumerate(binary_images)):
        if np.random.binomial(1, p):
            new_shape = (*image.shape, num_channels)
            bg = np.random.uniform(low, high, size=new_shape)
            fg = create_random_gradient_image(new_shape, low=low, high=high)
            three_channel_mask = np.stack([image] * num_channels, axis=-1)
            new_img = torch.as_tensor(((1 - three_channel_mask) * bg + three_channel_mask * fg)).permute(2, 0, 1)
        else:
            new_img = dataset.observations[index]
        augmented_dataset.observations.append(new_img)
    return augmented_dataset


def augment_background_textured(dataset: Dataset, texture_directory: str,
                                p: float = 0.0, p_empty: float = 0.0,
                                binary_images: List[np.ndarray] = None) -> Dataset:
    """
    parse background and fore ground. Give fore ground random color and background a crop of a textured image.
    :param binary_images:
    :param p_empty: in case of augmentation with texture, potentially augment without any foreground and empty action map (hacky!)
    :param p: probability for each image to be augmented with background
    :param dataset: augmented dataset in which observations are adjusted with new back- and foregrounds
    :param texture_directory: directory with sub-directories for each texture
    :return: augmented dataset
    """
    # 1. load texture image paths and keep in RAM
    if not texture_directory.startswith('/'):
        texture_directory = os.path.join(get_data_dir(os.environ['HOME']), texture_directory)
    texture_paths = [os.path.join(texture_directory, sub_directory, image)
                     for sub_directory in os.listdir(texture_directory)
                     if os.path.isdir(os.path.join(texture_directory, sub_directory))
                     for image in os.listdir(os.path.join(texture_directory, sub_directory))
                     if image.endswith('.jpg')]
    assert len(texture_paths) != 0
    # 2. parse binary images to extract fore- and background
    if binary_images is None:
        binary_images = parse_binary_maps(copy.deepcopy(dataset.observations), invert=True)
    augmented_dataset = copy.deepcopy(dataset)
    augmented_dataset.observations = []
    num_channels = dataset.observations[0].shape[0]
    for index, image in tqdm(enumerate(binary_images)):
        if np.random.binomial(1, p):
            new_shape = (*image.shape, num_channels)
            bg_image = np.random.choice(texture_paths)
            bg = load_and_preprocess_file(bg_image,
                                          size=(num_channels, image.shape[0], image.shape[1])).permute(1, 2, 0).numpy()
            if not np.random.binomial(1, p_empty):
                three_channel_mask = np.stack([image] * num_channels, axis=-1)
                fg = create_random_gradient_image(size=bg.shape)
                new_img = torch.as_tensor(((1 - three_channel_mask) * bg + three_channel_mask * fg)).permute(2, 0, 1)
                augmented_dataset.observations.append(new_img)
            else:
                augmented_dataset.observations.append(torch.as_tensor(bg).permute(2, 0, 1))
                augmented_dataset.actions[index] = torch.zeros_like(augmented_dataset.actions[index])
        else:
            augmented_dataset.observations.append(dataset.observations[index])
    return augmented_dataset


def create_random_gradient_image(size: tuple, low: float = 0., high: float = 1.) -> np.ndarray:
    color_a = np.random.uniform(low, high)
    color_b = np.random.uniform(low, high)
    locations = (np.random.randint(size[0]), np.random.randint(size[0]))
    while locations[0] == locations[1]:
        locations = (np.random.randint(size[0]), np.random.randint(size[0]))
    x1, x2 = min(locations), max(locations)
    gradient = np.arange(color_a, color_b, ((color_b - color_a)/(x2 - x1)))
    gradient = gradient[:x2-x1]
    assert len(gradient) == x2 - x1, f'failed: {len(gradient)} vs {x2 - x1}'
    vertical_gradient = np.concatenate([np.ones(x1) * color_a, gradient, np.ones(size[0] - x2) * color_b])
    image = np.stack([vertical_gradient]*size[1], axis=1).reshape(size)
    return image
