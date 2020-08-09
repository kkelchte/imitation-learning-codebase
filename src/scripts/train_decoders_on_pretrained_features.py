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

from src.ai.utils import mlp_creator
from src.data.utils import load_data_from_directory


class Parser(argparse.ArgumentParser):
    """Parser class to get config retrieve config file argument"""

    def __init__(self):
        super().__init__()
        self.add_argument("--output_path", type=str, default='/tmp/out')
        self.add_argument("--learning_rate", type=float, default=0.001)
        self.add_argument("--batch_size", type=int, default=64)
        self.add_argument("--training_epochs", type=int, default=100)
        self.add_argument("--task", type=str, default='normal')


if __name__ == '__main__':
    arguments = Parser().parse_args()

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
    filename = os.path.join(os.environ['DATADIR'], 'line_world_data', 'sim', f'noisy_augmented_3x256x256_0.hdf5')
    h5py_file = h5py.File(filename, 'r')

    ################################################################################
    # Define network                                                               #
    ################################################################################
    input_size = 8 * 16 * 16
    output_size = 64 * 64
    decoder = mlp_creator(sizes=[input_size, 2056, 2056, output_size],
                          activation=nn.ReLU,
                          output_activation=None,
                          bias_in_last_layer=False)

    ################################################################################
    # Take some training steps                                                     #
    ################################################################################

    total = len(h5py_file['dataset']['observations'])
    feature_type = 'normal'
    optimizer = torch.optim.Adam(params=decoder.parameters(), lr=arguments.learning_rate)
    losses = []
    for epoch in tqdm(range(arguments.training_epochs)):
        optimizer.zero_grad()
        sample_indices = np.random.choice(list(range(total)),
                                          size=arguments.batch_size)
        observations = [torch.as_tensor(h5py_file['dataset']['observations'][i], dtype=torch.float32) for i in
                        sample_indices]
        targets = [torch.as_tensor(h5py_file['dataset']['targets'][i], dtype=torch.float32)[:, ::2, ::2] for i in
                   sample_indices]
        representation = visualpriors.representation_transform(torch.stack(observations), feature_type, device='cpu')
        predictions = decoder(representation.view(-1, 2048)).view(-1, 64, 64)
        loss = (torch.stack(targets) - predictions).abs().sum()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        writer.write_scalar(loss.item(), 'training_loss')
        print(f'epoch {epoch}, loss: {loss.item()}')

    ################################################################################
    # Predict on real validation images                                            #
    ################################################################################
    validation_runs = [
        os.path.join(os.environ['DATADIR'], 'line_world_data', 'real', 'raw_data', d, 'raw_data', sd, 'observation')
        for d in
        ['concrete_bluecable', 'concrete_orangecable', 'concrete_whitecable', 'grass_bluecable', 'grass_orangecable']
        for sd in os.listdir(os.path.join(os.environ['DATADIR'], 'line_world_data', 'real', 'raw_data', d, 'raw_data'))]
    fig_counter = 0
    for run in validation_runs[0:1]:
        data = load_data_from_directory(run, size=(3, 256, 256))[1][::20]
        representation = visualpriors.representation_transform(torch.stack(data), feature_type, device='cpu')
        with torch.no_grad():
            predictions = decoder(representation.view(-1, 2048)).view(-1, 64, 64).detach().numpy()
        for pred in predictions:
            plt.imshow(pred)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'out', f'{fig_counter:010d}.jpg'))
            fig_counter += 1

    ################################################################################
    # Save checkpoint                                                              #
    ################################################################################
    torch.save({'state_dict': decoder.state_dict()},
               f'{output_path}/checkpoint.ckpt')
    print('done')
