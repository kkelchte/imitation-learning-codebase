import argparse
import os
import shutil

import h5py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import visualpriors
from visualpriors.taskonomy_network import TaskonomyDecoder
import subprocess

from src.ai.utils import mlp_creator
from src.data.utils import load_data_from_directory
from src.core.utils import get_date_time_tag

class Parser(argparse.ArgumentParser):
    """Parser class to get config retrieve config file argument"""

    def __init__(self):
        super().__init__()
        self.add_argument("--output_path", type=str, default='/tmp/out')
        self.add_argument("--learning_rate", type=float, default=0.001)
        self.add_argument("--batch_size", type=int, default=64)
        self.add_argument("--training_epochs", type=int, default=100)
        self.add_argument("--task", type=str, default='normal')
        self.add_argument("--loss", type=str, default='cross-entropy', help='options: cross-entropy, L1')
        self.add_argument("--mlp", action='store_true', help="use simple MLP instead of taskonomy decoder")



if __name__ == '__main__':
    arguments = Parser().parse_args()
    feature_type = arguments.task
    
    output_path = arguments.output_path if arguments.output_path.startswith('/') \
        else os.path.join(os.environ['DATADIR'], arguments.output_path)
    #if os.path.isdir(output_path):
    #    shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path, exist_ok=True)

    from src.core.tensorboard_wrapper import TensorboardWrapper
    writer = TensorboardWrapper(log_dir=output_path)

    ################################################################################
    # Load data                                                                    #
    ################################################################################
    print(f'{get_date_time_tag()}: load training data')
    filename = os.path.join(os.environ['DATADIR'], 'line_world_data', 'sim', f'noisy_augmented_3x256x256_0.hdf5')
    h5py_file = h5py.File(filename, 'r')

    ################################################################################
    # Define network                                                               #
    ################################################################################
    print(f'{get_date_time_tag()}: Define network')
    input_size = 8 * 16 * 16
    output_size = 64 * 64
    if arguments.mlp:
        decoder = mlp_creator(sizes=[input_size, 2056, 2056, output_size],
                              activation=nn.ReLU,
                              output_activation=None,
                              bias_in_last_layer=False)
    else:
        decoder = TaskonomyDecoder(out_channels=1,
                                    is_decoder_mlp=False,
                                    apply_tanh=True, 
                                    eval_only=False)
        for p in decoder.parameters():
            p.requires_grad = True

    ################################################################################
    # Take some training steps                                                     #
    ################################################################################
    print(f'{get_date_time_tag()}: Take {arguments.training_epochs} training steps')
    
    total = len(h5py_file['dataset']['observations']) - 100
    optimizer = torch.optim.Adam(params=decoder.parameters(), lr=arguments.learning_rate)
    losses = []
    for epoch in tqdm(range(arguments.training_epochs)):
        optimizer.zero_grad()
        sample_indices = np.random.choice(list(range(total)),
                                          size=arguments.batch_size)
        observations = [torch.as_tensor(h5py_file['dataset']['observations'][i], dtype=torch.float32) for i in
                        sample_indices]
        targets = torch.stack([torch.as_tensor(h5py_file['dataset']['targets'][0].repeat(2, axis=1).repeat(2, axis=2), dtype=torch.float32) for i in
                            sample_indices]).detach()
        
        representation = visualpriors.representation_transform(torch.stack(observations), feature_type, device='cpu')
        if arguments.mlp:
            predictions = decoder(representation.view(-1, 2048)).view(-1, 64, 64)
        else:
            predictions = (decoder(representation) + 1)/2
        assert predictions.min() > 0
        assert predictions.max() < 1

        if arguments.loss == 'L1':
            training_loss = (torch.stack(targets) - predictions[:, :, ::2, ::2]).abs().sum()
        else:
            training_loss =  (-targets * predictions.log() - (1 - targets) * (1 - predictions).log()).mean()
        
        training_loss.backward()
        optimizer.step()
        losses.append(training_loss.item())
        writer.write_scalar(training_loss.item(), 'training_loss')
        writer.increment_step()
        print(f'epoch {epoch}, loss: {training_loss.item()}')

    ################################################################################
    # Predict on training images                                                   #
    ################################################################################
    print(f'{get_date_time_tag()}: Predict on training images')
    
    fig_counter = 0
    output_dir = os.path.join(os.environ['TEMP'], 'out_train')
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    data = [torch.as_tensor(t, dtype=torch.float32) for t in h5py_file['dataset']['observations'][-100::2]]
    representation = visualpriors.representation_transform(torch.stack(data), feature_type, device='cpu')
    with torch.no_grad():
        if arguments.mlp:
            predictions = decoder(representation.view(-1, 2048)).view(-1, 64, 64).detach().numpy()
        else:
            predictions = ((decoder(representation) + 1)/2).view(-1, 256, 256).detach().numpy()

    for pred in tqdm(predictions):
        plt.imshow(pred)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'file{fig_counter:05d}.jpg'))
        fig_counter += 1

    if os.path.isfile(f'{output_path}/video_train.mp4'):
        os.remove(f'{output_path}/video_train.mp4')
    subprocess.call([
        'ffmpeg', '-framerate', '8', '-i', f'{output_dir}/file%05d.jpg', '-r', '30', '-pix_fmt', 'yuv420p',
        f'{output_path}/video_train.mp4'
    ])
    shutil.rmtree(output_dir) 
    data = None
    predictions = None

    ################################################################################
    # Predict on real validation images                                            #
    ################################################################################
    print(f'{get_date_time_tag()}: Predict on real validation images')
    validation_runs = [
        os.path.join(os.environ['DATADIR'], 'line_world_data', 'real', 'raw_data', d, 'raw_data', sd, 'observation')
        for d in
        ['concrete_bluecable', 'concrete_orangecable', 'concrete_whitecable', 'grass_bluecable', 'grass_orangecable']
        for sd in os.listdir(os.path.join(os.environ['DATADIR'], 'line_world_data', 'real', 'raw_data', d, 'raw_data'))]
    fig_counter = 0

    output_dir = os.path.join(os.environ['TEMP'], 'out_val')
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for run in validation_runs[::3]:
        data = load_data_from_directory(run, size=(3, 256, 256))[1][::10]
        representation = visualpriors.representation_transform(torch.stack(data), feature_type, device='cpu')
        with torch.no_grad():
            if arguments.mlp:
                predictions = decoder(representation.view(-1, 2048)).view(-1, 64, 64).detach().numpy()
            else:
                predictions = ((decoder(representation) + 1)/2).view(-1, 256, 256).detach().numpy()
            data = torch.stack(data).detach().numpy().swapaxes(1, 2).swapaxes(2, 3)
        for obs, pred in zip(data, predictions):
            if arguments.mlp:
                pred = pred.repeat(4, axis=0).repeat(4, axis=1)
            pred = np.stack([pred] * 3, axis=-1)
            img = 0.5 * obs + 0.5 * pred
            plt.imshow(img)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'file{fig_counter:05d}.jpg'))
            fig_counter += 1

    if os.path.isfile(f'{output_path}/video_val.mp4'):
        os.remove(f'{output_path}/video_val.mp4')
    subprocess.call([
        'ffmpeg', '-framerate', '8', '-i', f'{output_dir}/file%05d.jpg', '-r', '30', '-pix_fmt', 'yuv420p',
        f'{output_path}/video_val.mp4'
    ])
    shutil.rmtree(output_dir)

    ################################################################################
    # Save checkpoint                                                              #
    ################################################################################
    print(f'{get_date_time_tag()}: Save checkpoint')
    
    torch.save({'state_dict': decoder.state_dict()},
               f'{output_path}/checkpoint.ckpt')
    print(f'{get_date_time_tag()}: done')
