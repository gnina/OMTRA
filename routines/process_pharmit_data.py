import argparse

# TODO: this script should actually take as input just a hydra config 
# - but Ramith is setting up our hydra stuff yet, and we don't 
# yet know what the config for this dataset processing component will look like
# so for now just argparse, and once its written it'll be easy/concrete to 
# port into a hydra config
def parse_args():
    p = argparse.ArgumentParser(description='Process pharmit data')




    args = p.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args.input)
    print(args.output)