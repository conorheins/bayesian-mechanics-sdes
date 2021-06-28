import argparse
from jax import random


def parse_command_line(default_key = 0):
    """
    Wrapper for arg-parser used to get the input random seed and save mode
    from the user and set defaults if none is provided
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type = int,
                help = "key to initialize Jax random seed",
                dest = "init_key", default=default_key)
    parser.add_argument('--save', dest='save_mode', action='store_true')
    parser.add_argument('--no_save', dest='save_mode', action='store_false')
    parser.set_defaults(save_mode=True)

    args = parser.parse_args()

    key = random.PRNGKey(args.init_key) 
    save_mode = args.save_mode

    return key, save_mode