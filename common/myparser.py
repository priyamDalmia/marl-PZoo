import argparse

parser = argparse.ArgumentParser(prog="MARL-PZOO")
parser.add_argument("--env", default="LunarLander-v2")
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--time_steps", type=int, default=4)
parser.add_argument("--gamma", type=float, default=0.95)
parser.add_argument("--eps", type=float, default=1.0)
parser.add_argument("--eps_decay", type=float, default=0.995)
parser.add_argument("--eps_min", type=float, default=0.01)
parser.add_argument("--logdir", default="logs")


'''
Usage:
from myparser import parser
args = parser.parse_args



ogdir = os.path.join(

    args.logdir, parser.prog, args.env, \

    datetime.now().strftime("%Y%m%d-%H%M%S")

)

print(f"Saving training logs to:{logdir}")

writer = tf.summary.create_file_writer(logdir)
'''


