from argparse import ArgumentParser

PARSER = ArgumentParser()
PARSER.add_argument("file", help="The path to the dataset JSON file.")

GROUP = PARSER.add_mutually_exclusive_group(required=True)
GROUP.add_argument("-t", "--train", help="Train the specified model using the given dataset.", action="store_true")
GROUP.add_argument("-i", "--inference", help="Perform inference using the given model and dataset.", action="store_true")

GROUP = PARSER.add_mutually_exclusive_group(required=True)
GROUP.add_argument("-p", "--probabilistic", help="Use a probabilistic model.", action="store_true")
GROUP.add_argument("-n", "--neural", help="Use a neural network model.", action="store_true")
GROUP.add_argument("-o", "--other", help="Use a non-parametric model.", action="store_true")

ARGS = PARSER.parse_args()

if __name__ == '__main__':

    # TODO validate 'model' argument

    # TODO validate 'file' argument

    if ARGS.train:
        print("TODO: Run training code.")
    elif ARGS.inference:
        print("TODO: Run inference code.")
