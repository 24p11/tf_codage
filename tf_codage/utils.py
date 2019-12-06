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
            

def download_hdfs(input_file, output_file):
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