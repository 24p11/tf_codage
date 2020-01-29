from setuptools import setup, find_packages
setup(
    name="TensorFlow Codage",
    version="0.1",
    packages=['tf_codage'],
    install_requires=[
        'GPUtil',
        'matplotlib',
        'tensorflow>=2.0',
        'transformers>=2.2'],
    entry_points = {
        'console_scripts': ['download_hdfs_csv=tf_codage.cli.download_csv:main']
    }
)
