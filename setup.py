from setuptools import setup, find_packages
setup(
    name="TensorFlow Codage",
    version="1.0",
    packages=['tf_codage'],
    install_requires=[
        'click',
        'papermill',
        'GPUtil',
        'matplotlib',
        'tensorflow',
        'transformers',
        'scikit-learn',
        'pandas',
        'pyarrow'],
    entry_points = {
        'console_scripts': ['download_hdfs_csv=tf_codage.cli.download_csv:main',
                            'grep_keras_progress=tf_codage.cli.grep_keras_progress:main',
                            'train_model=tf_codage.cli.train_model:main']
    }
)
