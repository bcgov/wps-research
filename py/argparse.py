'''20230524 demo of argparse functionality (from argparse manual)

Used this in raster_warp_all.py
'''
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-v", "--verbose", action="store_true",
                    help="increase output verbosity")


parser.add_argument("square", type=int,
                    help="display a square of a given number")


args = parser.parse_args()
answer = args.square**2
if args.verbose:
    print(f"the square of {args.square} equals {answer}")
else:
    print(answer)
