import os
from shutil import copyfile
import yaml

import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import CubicSpline

# settings
from tqdm import tqdm

number_of_worlds = 1000
output_dir = 'line_worlds'
for world_index in tqdm(range(number_of_worlds)):
    number_of_points = np.random.randint(10, 30)
    dev = np.random.uniform(0.1, 0.5)


    # start from loop
    t = np.arange(0, 1., 1/number_of_points)
    clockwise = np.random.normal(0, 1) >= 0
    xo = np.cos(2*np.pi*t) - 1 if not clockwise else -np.cos(2*np.pi*t) + 1
    yo = np.sin(2*np.pi*t)

    # deviate points with noise
    x = [exo + np.sign(exo) * np.random.uniform(0, dev) * (i != 0 and i != 1 and i != (len(xo)-1) and i != (len(xo)-2))
         for i, exo in enumerate(xo)]
    y = [eyo + np.sign(eyo) * np.random.uniform(0, dev) * (i != 0 and i != 1 and i != (len(yo)-1) and i != (len(yo)-2))
         for i, eyo in enumerate(yo)]
    x += [x[0]]
    y += [y[0]]
    z = np.random.uniform(0.5, 2)

    waypoints = [[x[i], y[i], z] for i in range(len(x)-1)]
    tck, u = interpolate.splprep([x, y], s=0, k=3, per=True)
    unew = np.arange(0, 1.01, 1/(10*number_of_points))
    out = interpolate.splev(unew, tck)

    # Use interpolated points to connect tiny cylinders in gazebo
    r = 0.01
    l = 0.08

    # Load empty world:
    world_dir = 'src/sim/ros/gazebo/worlds'
    tree = ET.parse(os.path.join(os.environ['PWD'], world_dir, 'empty.world'))
    root = tree.getroot()
    world = root.find('world')

    # add model to world
    model = ET.SubElement(world, 'model', attrib={'name': 'line'})

    # Place small cylinders in one model
    x_coords, y_coords = out
    for index, (x, y) in enumerate(zip(x_coords, y_coords)):
        static = ET.SubElement(model, 'static')
        static.text = '1'
        link = ET.SubElement(model, 'link', attrib={'name': f'link_{index}'})
        pose = ET.SubElement(link, 'pose', attrib={'frame': ''})

        next_x = x_coords[(index + 1) % len(x_coords)]
        next_y = y_coords[(index + 1) % len(x_coords)]
        derivative = (next_y - y) / (next_x - x)
        slope = np.arctan(derivative)
        pose.text = f'{x} {y} {r} 0 1.57 {slope}'
        collision = ET.SubElement(link, 'collision', attrib={'name': 'collision'})
        visual = ET.SubElement(link, 'visual', attrib={'name': 'visual'})
        material = ET.SubElement(visual, 'material')
        script = ET.SubElement(material, 'script')
        name = ET.SubElement(script, 'name')
        name.text = 'Gazebo/Blue'
        uri = ET.SubElement(script, 'uri')
        uri.text = 'file://media/materials/scripts/gazebo.material'
        for element in [collision, visual]:
            geo = ET.SubElement(element, 'geometry')
            cylinder = ET.SubElement(geo, 'cylinder')
            radius = ET.SubElement(cylinder, 'radius')
            radius.text = str(r)
            length = ET.SubElement(cylinder, 'length')
            length.text = str(l)

    # Store world
#    model_name = f'model_{sum(x_coords)}'
    model_name = f'model_{world_index:03d}'
    world_dir = 'src/sim/ros/gazebo/worlds'
    os.makedirs(os.path.join(os.environ['PWD'], world_dir, output_dir), exist_ok=True)
    tree.write(os.path.join(os.environ['PWD'], world_dir, output_dir, model_name + '.world'), encoding="us-ascii",
               xml_declaration=True, method="xml")

    # Create world config with waypoints
    world_config_dir = 'src/sim/ros/config/world'
    background_file = f'src/sim/ros/gazebo/background_images/{output_dir}/{model_name}_{1 if clockwise else -1}_0_5.jpg'
    config = {
        'world_name': model_name,
        'max_duration': 300,
        'minimum_distance_px': 40,
        'max_distance_from_start': 10,
        'delay_evaluation': 1,
        'waypoints': [[float(w[0]), float(w[1]), float(w[2])] for w in waypoints],
        'waypoint_reached_distance': 0.2,
        'goal': {
            'x': {'min': float(waypoints[-3][0]) - 0.3,
                  'max': float(waypoints[-3][0]) + 0.3},
            'y': {'min': float(waypoints[-3][1]) - 0.3,
                  'max': float(waypoints[-3][1]) + 0.3},
            'z': {'min': 0.3,
                  'max': 1.8},
        },
        'background_file': background_file
    }

    os.makedirs(os.path.join(os.environ['PWD'], world_config_dir, output_dir), exist_ok=True)
    with open(os.path.join(os.environ['PWD'], world_config_dir, output_dir, model_name + '.yml'), 'w') as f:
        yaml.dump(config, f)

    # plot
    plt.figure(figsize=(11, 11))
    plt.plot(out[0], out[1])
    plt.ylim(-1.5, 1.5)
    if clockwise:
        plt.xlim(-0.5, 2.5)
    else:
        plt.xlim(-2.5, 0.5)
    plt.axis('off')

    os.makedirs(os.path.join(os.environ['PWD'], 'src/sim/ros/gazebo/background_images', output_dir), exist_ok=True)
    plt.savefig(os.path.join(os.environ['PWD'], background_file), bbox_inches='tight')
    plt.close()
    plt.cla()
    plt.clf()
print(f'finished: wrote {number_of_worlds}')
