"""Use papermill to run notebooks and save log + model files."""

import click
import papermill as pm
from papermill.cli import _resolve_type
from pathlib import Path
import os
import sys

from stat import S_IREAD, S_IRGRP, S_IROTH


@click.command()
@click.argument("notebook")
@click.option('-n', '--n-epochs', required=True, type=int,
    help="number of epochs (N_EPOCHS in notebook)")
@click.option('-p', '--parameter', 'cli_parameters', 
    required=False, multiple=True, nargs=2, type=str,
    help="set value of parameter PARAM to VALUE in the notebook, it can be used multiple times",
    metavar="PARAM VALUE")
@click.option('--save-path', default='../models/',
    help="path to save model artifacts (used to configure OUTPUT_DIR in the notebook)")
@click.option('--log-path', default='../logs/',
    help="path to save output notebook")
@click.option('-r', '--raw-parameter', 'raw_cli_parameters',
    required=False, multiple=True, nargs=2, type=str,
    help="like -p but do not convert the type")
@click.option('-P', '--silent-parameter', 'silent_cli_parameters',
    required=False, multiple=True, nargs=2, type=str,
    help="like -p but do not use the parameter in the filenames")
@click.option('-l', '--learning-rate', required=False, type=str,
    help="learning rate (LEARNING_RATE variable in notebook)")
@click.option('-s', '--suffix', required=False, type=str,
    help='suffix to set in file names')
@click.option('-f', '--force', is_flag=True,
    help='force ovewrite the log file')
@click.option('-g', '--gpu', default='auto', type=str,
    help='comma delimited list gpus to run the model on')
def main(
    notebook,
    n_epochs,
    learning_rate,
    cli_parameters,
    suffix,
    force,
    silent_cli_parameters, 
    raw_cli_parameters,
    save_path,
    gpu,
    log_path
):
    """Run NOTEBOOK from the command line and set parameters of notebook variables."""
    nb_parameters = dict(N_EPOCHS=n_epochs)
    if learning_rate:
        nb_parameters['LEARNING_RATE'] = float(learning_rate)
    for name, value in cli_parameters:
        nb_parameters[name] = _resolve_type(value)
    
    # these parameters should be considered string
    nb_parameters.update(raw_cli_parameters)
        
    core, _ = os.path.splitext(notebook)
    output_core = core + "-" + "-".join(
        "%s-%s" % (k.lower(), v) for k, v in nb_parameters.items())
    
    # add suffix to avoid overwriting log files
    if suffix:
        output_core += "-" + suffix
 
    log_path =  str(Path(log_path) / (output_core + ".ipynb"))
    output_dir = str(Path(save_path) / output_core)
    
    if os.path.isfile(log_path):
        if not force:
            print("Log file %s already exists. Consider passing --suffix argument." % log_path)
            sys.exit(1)
        else:
            print("Log file %s exists. Overwrite forced with option -f/--force." % log_path)
    
    if gpu:
        if gpu == 'auto':
            available_devices = str(GPUtil.getFirstAvailable(maxMemory=0.1)[0])
            assert available_devices, "no free GPU found, if you want to run on CPU use `--gpu none`"
        elif gpu == 'none':
            available_devices = ""
        else:
            available_devices = gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", available_devices)  
        print("using GPUs:", available_devices)
    
    # these params should not be present in file name
    nb_parameters['VERBOSITY'] = 2
    nb_parameters['OUTPUT_DIR'] = output_dir
    nb_parameters.update(silent_cli_parameters)
        
    print("training")
    print("Saving log to:", log_path)
    print("Saving model data to:", output_dir)
    pm.execute_notebook(
        notebook,
        log_path,
        parameters=nb_parameters,
    )

    print("Locking log file", log_path)
    os.chmod(log_path, S_IREAD|S_IRGRP|S_IROTH)