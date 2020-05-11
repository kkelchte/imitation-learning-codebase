#!/usr/bin/python3.7
import os

from graphviz import Digraph

from src.core.utils import get_to_root_dir


def parse_dependencies(file_path: str) -> list:
    module_dependencies = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for l in lines:
        if 'from src.' in l:
            l = l.strip()
            total_module = l.split(' ')[1]
            module = '.'.join(total_module.split('.')[1:])
            if module in python_files.keys():
                module_dependencies.append(module)
    return module_dependencies


get_to_root_dir()
root = os.getcwd()
destination = os.path.join(root, 'dependency-graph')
os.makedirs(destination, exist_ok=True)
python_files = {}
exclude_dirs = ['ros', 'algorithms', 'test', 'catkin_generated', 'atomic_configure', 'devel',
                'installspace', 'rosnodes', 'architectures', 'core']
exclude_files = ['data_cleaning.py', 'utils.py']

for dirpath, dnames, fnames in os.walk(os.path.join(root, 'src')):
    if os.path.basename(dirpath) in exclude_dirs:
        continue
    for f in fnames:
        if f.endswith(".py") and f not in exclude_files:
            module = 'src.'.join(dirpath.split('src')[1:])
            module += f'.{f[:-3]}'
            module = module.replace('/', '.')
            module = module.replace('..', '.')
            module = module[1:]
            python_files[module] = os.path.join(dirpath, f)
for f in sorted(python_files.keys()):
    print(f)

dependencies = {}
for module in sorted(python_files.keys()):
    dependencies[module] = parse_dependencies(python_files[module])

g = Digraph('G', filename=os.path.join(destination, 'dependency-graph'))
for node in sorted(dependencies.keys()):
    for edge in sorted(dependencies[node]):
        g.edge(node, edge)
g.view()




