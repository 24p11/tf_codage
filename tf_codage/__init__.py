import os
import GPUtil
os.environ['CUDA_DEVICE_ORDER'] = os.environ.get('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
available_devices = str(GPUtil.getFirstAvailable(maxMemory=0.1)[0])
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', available_devices)

def print_env():
    """Dump environment to the standard output"""
    for k, v in os.environ.items():
        print('{}={}'.format(k, v))