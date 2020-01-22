import sys
from contextlib import redirect_stdout

class TeeStream:
    """Redirect stream to many different outputs.
    
    This is required so that we can see the progress bar 
    both in the notebook and in the terminal (for runs
    from command line)"""
    
    def __init__(self, *streams):
        self._streams = streams
    
    def write(self, data):
        for s in self._streams:
            s.write(data)
            
    def flush(self):
        for s in self._streams:
            s.flush()
            
def notebook_copy_stdout():
    """Context manager to copy the notebook output to standard output (terminal)"""
    tee = TeeStream(sys.stdout, sys.__stdout__)
    redirect = redirect_stdout(tee)
    return redirect

def download_hdfs(input_file, output_file):
    """Download files from HDFS store."""
    import sys
    import pyarrow
    conn = pyarrow.hdfs.connect()

    output_file = sys.argv[1]
    input_file = sys.argv[1]

    filenames = conn.ls(input_file)
    filenames = [f for f in filenames if f.endswith('.csv')]
    print(filenames)

    output_file = open(output_file, 'wb')
    for f in filenames:
        f = f.replace('hdfs://bbsedsi', '')
        print("Downloading {}".format(f))
        fid = conn.open(f)
        output_file.write(fid.read())
    output_file.close()
    
def save_model(model, model_name, metric_data=None, tokenizer=None, encoder=None, root_dir='..'):
    """Save model to subdirs of `root_dir` together with the metrics, encoder and tokenzier if specified."""

    import os

    model_dir = os.path.join(root_dir, 'models', model_name)
    print('Saving model to ', model_dir)

    os.makedirs(model_dir, exist_ok=True)

    model.save_pretrained(model_dir)

    if metric_data:
        
        metrics_dir = os.path.join(root_dir, 'metrics') 
        os.makedirs(metrics_dir, exist_ok=True)
        

        metric_dict = dict(zip(model.metrics_names,
                 map(float, metric_data)))

        import json
        with open(
            os.path.join(
                metrics_dir,
                model_name + '.json'), 'w') as fid:
            json.dump(metric_dict, fid)


    if tokenizer:
        tokenizer.save_pretrained(model_dir)

    if encoder:
        import joblib
        joblib.dump(encoder, os.path.join(model_dir, 'encoder.joblib'))
    
