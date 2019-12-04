import os
os.environ['CUDA_DEVICE_ORDER'] = os.environ.get('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '2')

def print_env():
    """Dump environment to the standard output"""
    for k, v in os.environ.items():
        print(f'{k}={v}')