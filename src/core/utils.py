#!/usr/python
import glob
import os
import subprocess
from datetime import datetime
from typing import List

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_data_dir(alternative: str) -> str:
    """
    Check if DATADIR is an environment variable, otherwise return provided alternative
    :param alternative: alternative to be returned in case no datadir was provided
    :return: Data directory
    """
    return os.environ['DATADIR'] if 'DATADIR' in os.environ.keys() else alternative


def get_file_length(file_path: str) -> int:
    with open(file_path, 'r') as f:
        return len(f.readlines())


def get_check_sum_list(data: List[np.ndarray]) -> float:
    check_sum = 0
    for e in data:
        check_sum += np.sum(np.asarray(e))
    return check_sum


def read_file_to_output(file_path: str) -> None:
    print('#' * 50 + ' ' * 5 + os.path.basename(file_path) + ' ' * 5 + '#' * 50)
    with open(file_path, 'r') as f:
        for line in f.readlines():
            print(line.strip())
    print('#' * 50 + ' ' * 5 + 'END' + ' ' * 5 + '#' * 50)


def camelcase_to_snake_format(input: str) -> str:
    output = ''
    prev_c = ''
    for c in input:
        if c.isupper():
            output += '_' + c.lower() if prev_c.isalpha() else c.lower()
        else:
            output += c
        prev_c = c
    return output


def get_filename_without_extension(filename: str) -> str:
    return str(os.path.basename(filename).split('.')[0])


def get_date_time_tag() -> str:
    return str(datetime.strftime(datetime.now(), format="%y-%m-%d_%H-%M-%S"))


def count_grep_name(grep_str: str) -> int:
    ps_process = subprocess.Popen(["ps", "-ef"],
                                  stdout=subprocess.PIPE)
    with ps_process.stdout:
        grep_process = subprocess.Popen(["grep", grep_str],
                                        stdin=ps_process.stdout,
                                        stdout=subprocess.PIPE)
        with grep_process.stdout:
            output_string = str(grep_process.communicate()[0])
    processed_output_string = [line for line in output_string.split('\\n') if 'grep' not in line
                               and 'test' not in line and len(line) > len(grep_str) and 'pycharm' not in line]
    return len(processed_output_string)


def get_to_root_dir():
    # assume you're in a subfolder in the codebase:
    while 'ROOTDIR' not in os.listdir('.'):
        os.chdir('..')
        if os.getcwd() == '/':
            raise FileNotFoundError


def generate_random_image(size: tuple) -> np.ndarray:
    return np.random.randint(0, 255, size=size, dtype=np.uint8)


def tensorboard_write_distribution(writer, distribution, tag, step) -> None:
    writer.add_scalar(f"{tag} mean", distribution.mean, global_step=step)
    writer.add_scalar(f"{tag} std", distribution.std, global_step=step)

#######################################
# Extensive evaluation helper functions
#######################################


def save_output_plots(output_dir: str, data: dict) -> None:
    """
    Creates an overview of different arrays stored in data dictionary with key used for legend
    :param output_dir: location where output_plot.jpg is saved
    :param data: dictionary with key: val pairs e.g. 'expert': np.array((6, 100)), 'network': np.array((6, 100))
    :return: None
    """
    title_dictionary = {
        1: ['CMD'],
        2: ['SPEED', 'TURN'],
        4: ['linear X', 'linear Y', 'linear Z', 'angular Z'],
        6: ['linear X', 'linear Y', 'linear Z', 'angular X', 'angular Y', 'angular Z']
    }
    colors = [f'C{i}' for i in range(10)]
    data_shape = list(data.values())[0].shape
    f, axes = plt.subplots(data_shape[-1], 1,
                           sharex=True, sharey=True, figsize=(20, 3*data_shape[-1]))
    for ax_index, ax in enumerate(axes):
        ax.set_title(title_dictionary[data_shape[-1]][ax_index] if data_shape[-1] in title_dictionary.keys()
                     else str(ax_index))
        for key_index, data_key in enumerate(sorted(data.keys())):
            ax.plot([d[ax_index] for d in data[data_key]], colors[key_index], label=data_key)
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'output_plot.jpg'))


def create_output_video(output_dir: str, observations: list, actions: dict) -> None:
    """
    Creates a video where actions are annotated on frames.
    :param output_dir: location where output_plot.jpg is saved
    :param observations: list of images
    :param actions: dictionary with key: val pairs e.g. 'expert': np.array((6, 100)), 'network': np.array((6, 100))
    :return: None
    """
    folder = os.path.join(output_dir, 'video')
    os.makedirs(folder, exist_ok=True)

    for image_index in tqdm(range(len(observations)), ascii=True, desc='video'):
        plt.figure()
        border_width = 100
        image = observations[image_index].squeeze()

        image = np.stack((image,) * 3, axis=-1)
        border = np.ones((image.shape[0], border_width, 3), dtype=image.dtype)
        image = np.concatenate([border, image], axis=1)

        colors = [(190 / 255., 114 / 255., 64 / 255.),
                  (75 / 255., 150 / 255., 200 / 255.)]

        for index, (label, value) in enumerate(actions.items()):
            forward_speed = 100 * (value[image_index][0] + 0.1)
            direction = np.arccos(value[image_index][-1])
            origin = (50, int(image.shape[0] / 2))
            steering_point = (int(origin[0] - forward_speed * np.cos(direction)),
                              int(origin[1] - forward_speed * np.sin(direction)))
            image = cv2.arrowedLine(image, origin, steering_point, color=colors[index], thickness=1)
            image = cv2.putText(image, label, (3, origin[1] + 55 + index * 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[index], thickness=1)
        image = cv2.circle(image, origin, radius=20, color=(0, 0, 0, 0.3), thickness=1)

        plt.tight_layout()
        plt.axis('off')
        plt.imshow(image)
        plt.savefig(folder + "/file%02d.jpg" % image_index)
        plt.close()
        plt.cla()

    original_directory = os.getcwd()
    os.chdir(folder)
    subprocess.call([
        'ffmpeg', '-framerate', '8', '-i', 'file%02d.jpg', '-r', '30', '-pix_fmt', 'yuv420p',
        'video.mp4'
    ])
    for file_name in glob.glob("*.jpg"):
        os.remove(file_name)
    os.chdir(original_directory)
