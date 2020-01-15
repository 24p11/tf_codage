from setuptools import setup, find_packages
setup(
    name="TensorFlow Codage",
    version="0.1",
    packages=['tf_codage'],
    entry_points = {
        'console_scripts': ['download_hdfs_csv=tf_codage.cli.download_csv:main']
    }
)
