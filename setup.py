from setuptools import setup

setup(
    # other arguments here...
    entry_points={
        'console_scripts': [
            'sim1 = example.medication_adherence_sim1:main',
        ]
    }
)
