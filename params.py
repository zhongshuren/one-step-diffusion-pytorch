import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--num_eval', default=1024, type=int)
parser.add_argument('--lr', default='3e-4', type=float)
