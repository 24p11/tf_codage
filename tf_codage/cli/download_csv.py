import argparse
import sys
import pyarrow

parser = argparse.ArgumentParser(
    description="Download spark-generated mulit-part CSV from HDFS and stich them together"
)
parser.add_argument(
    "input_file",
    help="path on the HDFS file system, it must point to a directory with multiple CSV files",
)
parser.add_argument(
    "output_file",
    nargs="?",
    help="path to save the final CSV file to, prints to the terminal if not defined",
)
parser.add_argument(
    "--has_header",
    default=False,
    action="store_true",
    help="interpret first line of each CSV as header",
)


def main():
    args = parser.parse_args()

    conn = pyarrow.hdfs.connect()

    input_file = args.input_file

    # if output not given use the name of input file but store locally
    if not args.output_file:
        output_file = args.input_file
    else:
        output_file = args.output_file
    has_header = args.has_header

    filenames = conn.ls("/user/cse170020/bartosz_csv/" + input_file)
    filenames = [f for f in filenames if f.endswith(".csv")]

    output_file = open(output_file, "wb")

    # copy header to output if header defined
    # first file is copied completely
    # if the file is empty move to the next one
    header = ""
    i = 0
    while not header:
        print("Downloading {}".format(filenames[i]))
        fid = conn.open(filenames[i])
        header = fid.read()
        fid.close()
        i += 1
    output_file.write(header)

    for f in filenames[i:]:
        print("Downloading {}".format(f))
        fid = conn.open(f)
        buffer = fid.read()
        if has_header:
            lines = buffer.split(b"\n")
            buffer = b"\n".join(lines[1:])
        output_file.write(buffer)
        fid.close()
    output_file.close()
