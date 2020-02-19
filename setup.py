from setuptools import setup, find_packages
setup(
    name="TensorFlow Codage",
    version="0.1",
    packages=['tf_codage'],
    install_requires=[
        'GPUtil',
        'matplotlib',
        'tensorflow>=2.0',
        'transformers==2.3.0',
        'scikit-learn>=0.21.3',
        'pandas>=0.25.3',
        'pyarrow>=0.15.1'],
    entry_points = {
        'console_scripts': ['download_hdfs_csv=tf_codage.cli.download_csv:main',
                            'grep_keras_progress=tf_codage.cli.grep_keras_progress:main']
    }
)
