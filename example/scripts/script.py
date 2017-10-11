import os
from datetime import datetime
import argparse
import sys
sys.path.append("../")
sys.path.append("../../")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bandit.agent import ContextualAgent
from bandit.bandit import MultiBandits
from bandit.policy import LinUCBPolicy, RandomPolicy
from envs.bandits import Patient, PatientEngagement
from envs.envs import PatientEnv2


def main(args):
    patients = np.vstack(
        (np.tile([1, 0, 0], (20, 1)),
         np.tile([0, 1, 0], (20, 1)),
         np.tile([1, 0, 1], (20, 1))))
    k = 3  # NUmber of arms
    d = 3

    d = d + 1 if args.use_intercept else d

    bandits = []
    for patient_id, barriers in enumerate(patients):
        bandits.append(
            Patient(k, d, barriers=barriers, patient_id=patient_id,
                    prev=args.lookback, use_intercept=args.use_intercept))  # reshape=args.reshape))

    env = PatientEnv2(args.iterations, args.epochs)
    env.make_env(bandits)
    policy = LinUCBPolicy(args.alpha, d)
    agent = ContextualAgent(k, d, policy,
                            init_exploration=args.init_exploration*len(bandits))

    env.run(agent)

    if args.save:
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)

        env.save_result(alpha=args.alpha, epochs=args.epochs,
                        filename=args.filename,
                        save_path=args.save_path)
    if args.interactive:
        from IPython import embed
        embed()
    # env.plot_adherence(args.alpha)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument('--init_exploration',
                        help='Periods of initial exploration',
                        default=10,
                        type=int
                        )
    parser.add_argument('--lookback',
                        help='lookback period (state for patient)',
                        default=5,
                        type=int
                        )
    parser.add_argument('--iterations',
                        help='How many periods to simulate this environment',
                        default=200,
                        type=int
                        )
    parser.add_argument('--epochs',
                        help='How many epochs to simulate this environment',
                        default=1,
                        type=int
                        )
    parser.add_argument('--alpha',
                        help='Hyperparameter for linUCB',
                        default=0.1,
                        type=float
                        )
    parser.add_argument('--save_path',
                        default='./results',
                        help='Result dataframe save path'
                        )
    parser.add_argument('--save',
                        action='store_true'
                        )
    parser.add_argument('--interactive',
                        action='store_true'
                        )
    parser.add_argument('--reshape',
                        action='store_true'
                        )
    parser.add_argument('--no-reshape',
                        action='store_false'
                        )
    parser.add_argument('--filename',
                        help='Filename'
                        )
    parser.add_argument('--use_intercept',
                         action='store_true'
                         )
    parser.add_argument('--no-use_intercept',
                        action='store_false'
                        )

    args, unknown = parser.parse_known_args()
    main(args)
