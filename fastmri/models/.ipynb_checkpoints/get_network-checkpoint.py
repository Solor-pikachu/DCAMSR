import hashlib
import os
import torch
import importlib
import timm
def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.
    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.
    Returns:
        A generator for all the interested files with relative pathes.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = os.path.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(
                        entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)

arch_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'archs')
arch_filenames = [
    os.path.splitext(os.path.basename(v))[0] for v in scandir(arch_folder)
    if v.endswith('_arch.py')
]
_arch_modules = [
    importlib.import_module('fastmri.models.archs.{}'.format(file_name))
    for file_name in arch_filenames
]

def dynamic_instantiation(modules, net_name,scale):
    cls_type = net_name
    cls_ = None
    for module in modules:
        cls_ = getattr(module, cls_type, None)
        if cls_ is not None:
            break
    if cls_ is None:
        raise ValueError('{} is not found.'.format(cls_type))
    return cls_(None,scale=scale)


def define_network(net_name,scale):
    net = dynamic_instantiation(_arch_modules, net_name,scale)
    return net