from setuptools import setup, find_packages

setup_requires = [
    ]

install_requires = [
    # 'numpy==1.14.0',
    # 'pandas==0.22.0',
    # 'torch==0.3.0.post4',
    # 'torchvision==0.2.0',
    # 'tqdm==4.19.4',
    'numpy',
    'pandas',
    'torch',
    'torchvision',
    'tqdm'
    ]

dependency_links = [
    ]

setup(
    name                =   'pygan',
    version             =   '0.0.1',
    description         =   'GAN-based Data Generator In Python 3',
    author              =   'Giyeom Kim',
    author_email        =   'gompu123@gmail.com',
    url                 =   'https://github.com/gompu123/pygan',
    install_requires    =   install_requires,
    packages            =   find_packages(exclude = ['docs', 'example']),
    keywords            =   [ 'Machine Learning', 'Deep Learning', 'AI', 'GAN', 'Generative Adversarial Networks', 'Data Augmentation'],
    setup_requires      =   setup_requires,
    dependency_links    =   dependency_links,
    classifiers         =   [
        'Programming Language :: Python :: 3.6'
    ]
)
