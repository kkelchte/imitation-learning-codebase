import argparse
import copy
import json
import os
import shutil
from glob import glob

import h5py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from visualpriors.taskonomy_network import TaskonomyDecoder
import subprocess
from visualpriors.transforms import VisualPriorRepresentation

from src.ai.utils import mlp_creator
from src.data.utils import load_data_from_directory
from src.core.utils import get_date_time_tag


class Parser(argparse.ArgumentParser):
    """Parser class to get config retrieve config file argument"""

    def __init__(self):
        super().__init__()
        self.add_argument("--output_path", type=str, default='/tmp/out')
        self.add_argument("--dataset", type=str, default='noisy_augmented')
        self.add_argument("--datadir", type=str, default=None)
        self.add_argument("--learning_rate", type=float, default=0.001)
        self.add_argument("--batch_size", type=int, default=2)
        self.add_argument("--training_epochs", type=int, default=1)
        self.add_argument("--task", type=str, default='normal')
        self.add_argument("--loss", type=str, default='cross-entropy', help='options: cross-entropy, L1')
        self.add_argument("--mlp", action='store_true', help="use simple MLP instead of taskonomy decoder")
        self.add_argument("--end_to_end", action='store_true', help="allow encoder net to change (with lower LR)")
        self.add_argument("--side_tuning", action='store_true', help="allow one encoder net to change")
        self.add_argument("--skip_sim_video", action='store_true', help="skip video creation")
        self.add_argument("--skip_real_video", action='store_true', help="skip video creation")


if __name__ == '__main__':
    arguments = Parser().parse_args()
    feature_type = arguments.task

    local_output_path = os.path.join(f'{os.environ["TEMP"]}/train_decoders')
    from src.core.tensorboard_wrapper import TensorboardWrapper
    writer = TensorboardWrapper(log_dir=local_output_path)

    ################################################################################
    # Load data                                                                    #
    ################################################################################
    print(f'{get_date_time_tag()}: load training data')
    datadir = os.environ['DATADIR'] if arguments.datadir is None else arguments.datadir
    filename = os.path.join(datadir, 'line_world_data', 'sim', f'{arguments.dataset}_3x256x256_0.hdf5')
    h5py_file = h5py.File(filename, 'r')

    ################################################################################
    # Define network                                                               #
    ################################################################################
    print(f'{get_date_time_tag()}: Define network')
    input_size = 8 * 16 * 16
    output_size = 64 * 64

    VisualPriorRepresentation._load_unloaded_nets([feature_type])
    encoder = VisualPriorRepresentation.feature_task_to_net[feature_type]
    fixed_encoder = copy.deepcopy(encoder) if arguments.side_tuning else None
    
    if arguments.end_to_end or arguments.side_tuning:
        for p in encoder.parameters():
            p.requires_grad = True

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
    encoder_optimizer = torch.optim.Adam(params=encoder.parameters(), lr=arguments.learning_rate/10)
    optimizer = torch.optim.Adam(params=decoder.parameters(), lr=arguments.learning_rate)
    losses = []

    validation_observations = torch.stack([torch.as_tensor(t, dtype=torch.float32) for t in h5py_file['dataset']['observations'][-100::2]])
    validation_targets = torch.stack([torch.as_tensor(t, dtype=torch.float32) for t in h5py_file['dataset']['targets'][-100::2]])
        
    for epoch in tqdm(range(arguments.training_epochs)):
        optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        sample_indices = np.random.choice(list(range(total)),
                                          size=arguments.batch_size)
        observations = torch.stack([torch.as_tensor(h5py_file['dataset']['observations'][i], dtype=torch.float32) for i in
                                    sample_indices])
        targets = torch.stack([torch.as_tensor(h5py_file['dataset']['targets'][i],
                                               dtype=torch.float32) for i in sample_indices]).detach()

#        representation = visualpriors.representation_transform(torch.stack(observations), feature_type, device='cpu')
        representation = encoder(observations)
        if arguments.side_tuning:
            fixed_representation = fixed_encoder(observations)
            representation += fixed_representation
        if arguments.mlp:
            predictions = decoder(representation.view(-1, 2048)).view(-1, 64, 64)
        else:
            predictions = (decoder(representation) + 1)/2

        if arguments.loss == 'L1':
            training_loss = (targets - predictions[:, :, ::2, ::2]).abs().mean()
        else:
            training_loss = (-targets * predictions.log() - (1 - targets) * (1 - predictions).log()).mean()
        training_loss.backward()
        optimizer.step()
        if arguments.end_to_end or arguments.side_tuning:
            encoder_optimizer.step()
        
        losses.append(training_loss.item())
        writer.write_scalar(training_loss.item(), 'training_loss')
        
        # calculate validation loss
        representation = encoder(validation_observations)
        if arguments.side_tuning:
            fixed_representation = fixed_encoder(validation_observations)
            representation += fixed_representation
        if arguments.mlp:
            predictions = decoder(representation.view(-1, 2048)).view(-1, 64, 64)
        else:
            predictions = (decoder(representation) + 1)/2
        validation_loss = (validation_targets - predictions).abs().mean()
        writer.write_scalar(validation_loss.item(), 'validation_loss')
        writer.increment_step()
        print(f'epoch {epoch}, training loss: {training_loss.item()}, validation loss: {validation_loss.item()}')

    ################################################################################
    # Predict on simulated images                                                   #
    ################################################################################
    if not arguments.skip_sim_video:
        print(f'{get_date_time_tag()}: Predict on simulated images')
        
        fig_counter = 0
        output_dir = os.path.join(local_output_path, 'out_train')
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        data = [torch.as_tensor(t, dtype=torch.float32) for t in h5py_file['dataset']['observations'][-100::2]]
        representation = encoder(torch.stack(data))
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

        data = None
        predictions = None

    ################################################################################
    # Predict on real validation images                                            #
    ################################################################################
    if not arguments.skip_real_video:
        print(f'{get_date_time_tag()}: Predict on real validation images')

        validation_runs = [
            os.path.join(datadir, 'line_world_data', 'real', 'raw_data', d, 'raw_data', sd, 'observation')
            for d in
            ['concrete_bluecable', 'concrete_orangecable', 'concrete_whitecable', 'grass_bluecable', 'grass_orangecable']
            for sd in os.listdir(os.path.join(datadir, 'line_world_data', 'real', 'raw_data', d, 'raw_data'))]
        fig_counter = 0

        output_dir_val = os.path.join(local_output_path, 'out_val')
        if os.path.isdir(output_dir_val):
            shutil.rmtree(output_dir_val)
        os.makedirs(output_dir_val, exist_ok=True)

        for run in tqdm(validation_runs[::3]):
            data = load_data_from_directory(run, size=(3, 256, 256))[1][::20]
            representation = encoder(torch.stack(data))
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
                img = obs * pred
                plt.imshow(img)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir_val, f'file{fig_counter:05d}.jpg'))
                plt.close()
                plt.cla()
                fig_counter += 1

    ################################################################################
    # Save checkpoint      (mount opal the latest)                                 #
    ################################################################################
    print(f'{get_date_time_tag()}: Save checkpoint')

    output_path = arguments.output_path if arguments.output_path.startswith('/') \
        else os.path.join(os.environ['DATADIR'], arguments.output_path)
    os.makedirs(output_path, exist_ok=True)
    
    if not arguments.skip_sim_video:
        if os.path.isfile(f'{output_path}/video_train.mp4'):
            os.remove(f'{output_path}/video_train.mp4')
        subprocess.call([
            'ffmpeg', '-framerate', '8', '-i', f'{output_dir}/file%05d.jpg', '-r', '30', '-pix_fmt', 'yuv420p',
            f'{output_path}/video_train.mp4'
        ])
        shutil.rmtree(output_dir) 

    if not arguments.skip_real_video:
        if os.path.isfile(f'{output_path}/video_val.mp4'):
            os.remove(f'{output_path}/video_val.mp4')
        subprocess.call([
            'ffmpeg', '-framerate', '8', '-i', f'{output_dir_val}/file%05d.jpg', '-r', '30', '-pix_fmt', 'yuv420p',
            f'{output_path}/video_val.mp4'
        ])
        shutil.rmtree(output_dir_val)

    tensorboard_file = glob(f'{local_output_path}/events.*')[0]
    os.system(f'mv {tensorboard_file} {output_path}/{os.path.basename(tensorboard_file)}')

    torch.save({'decoder': {'state_dict': decoder.state_dict()},
                'encoder': {'state_dict': encoder.state_dict()}},
               f'{output_path}/checkpoint.ckpt')

    with open(os.path.join(output_path, 'config.json'), 'w') as f:
        json.dump(arguments.__dict__, f)

    shutil.rmtree(local_output_path)
    print(f'{get_date_time_tag()}: done')
