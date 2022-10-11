from distutils.core import setup

setup(name='Dynamic_GNN_structured_models',
      version='1.0.0',
      install_requires=[
            'hydra-core==1.0', #hydra-core 1.1 has some congif_path issues
            # 'torch',
            # 'torch-geometric==1.7.2', # recent versions do batching differently
            'plotly',
            # 'wandb',
            'pyquaternion',
            # 'pytorch_lightning',
            'shapely',
            'async-savers',
            'graphviz',
            'filelock',
            'gym',
            'matplotlib',
            'stable-baselines3[extra]',
            'ipdb'
      ],
      description='Dynamic Inference on Graphs using Structured Transition Models',
      author='Saumya Saxena',
      author_email='saumyas@andrew.cmu.edu',
      url='none',
      packages=['Dynamic_GNN_structured_models']
     )