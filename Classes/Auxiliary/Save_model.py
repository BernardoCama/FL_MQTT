from Classes.Params.param import NUM_ROUNDS
import sys
import numpy as np
import _pickle as cPickle
import os
import shutil

def Save_model(WEIGHTS, client_, cwd, round):

    DIR_WEIGHTS = os.path.join(cwd, 'H{}'.format(client_),  'saved_weights')

    # Create folder of weights
    if not os.path.exists(DIR_WEIGHTS):
        os.makedirs(DIR_WEIGHTS)

    # Remove folder of this round
    round_dir = os.path.join(DIR_WEIGHTS, '{}'.format(round))
    if os.path.exists(round_dir):
        shutil.rmtree(round_dir)

    rounds = next(os.walk(DIR_WEIGHTS))[1]
    rounds.sort(key=float)
    rounds_dir = [os.path.join(next(os.walk(DIR_WEIGHTS))[0], s) for s in rounds]

    # Save weights of last n rounds
    if len(rounds_dir) >= 100:
        shutil.rmtree(rounds_dir[0])

    os.makedirs(round_dir)
    np.save(os.path.join(round_dir, 'weights.npy'), WEIGHTS, allow_pickle = True)
  