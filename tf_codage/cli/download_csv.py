import argparse
import sys
import pyarrow

def main():
    conn = pyarrow.hdfs.connect()

    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('output_file', nargs='?')
    parser.add_argument('--has_header', default=False, action='store_true')
    args = parser.parse_args()

    input_file = args.input_file

    # if output not given use the name of input file but store locally
    if not args.output_file:
        output_file = args.input_file
    else:
        output_file = args.output_file
    has_header = args.has_header

    filenames = conn.ls('/user/cse170020/bartosz_csv/' + input_file)
    filenames = [f for f in filenames if f.endswith('.csv')]

    output_file = open(output_file, 'wb')
    
    # copy header to output if header defined
    # first file is copied completely
    print("Downloading {}".format(filenames[0]))
    fid = conn.open(filenames[0])
    header = fid.read()
    output_file.write(header)
    fid.close()

    for f in filenames[1:]:
        print("Downloading {}".format(f))
        fid = conn.open(f)
        buffer = fid.read()
        if has_header:
            lines = buffer.split(b'\n')
            buffer = b"\n".join(lines[1:])
        output_file.write(buffer)
        fid.close()
    output_file.close()
