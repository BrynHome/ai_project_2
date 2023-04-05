from argparse import ArgumentParser
from me_probabilistic import probabilistic_predict, probabilistic_train
from sw import sw_train, sw_predict
from bh import bh_train, bh_predict

PARSER = ArgumentParser()
PARSER.add_argument("file", help="The path to the dataset JSON file.")

GROUP = PARSER.add_mutually_exclusive_group(required=True)
GROUP.add_argument("-t", "--train", help="Train the specified model using the given dataset.", action="store_true")
GROUP.add_argument("-i", "--inference", help="Perform inference using the given model and dataset.", action="store_true")

GROUP = PARSER.add_mutually_exclusive_group(required=True)
GROUP.add_argument("-p", "--probabilistic", help="Use a probabilistic model.", action="store_true")
GROUP.add_argument("-n", "--neural", help="Use a neural network model.", action="store_true")
GROUP.add_argument("-o", "--other", help="Use a non-parametric model.", action="store_true")

PARSER.add_argument("-f", "--feature-selection", help="Enable feature selection for the probablistic model", dest="feature_selection", action="store_true")

ARGS = PARSER.parse_args()

if __name__ == '__main__':

    # TODO validate 'model' argument

    # TODO validate 'file' argument

    if ARGS.train:
        if ARGS.probabilistic:
            probabilistic_train(ARGS.file, feature_selection=ARGS.feature_selection)
        if ARGS.neural:
            bh_train(ARGS.file)
        if ARGS.other:
            sw_train(ARGS.file)
    elif ARGS.inference:
        if ARGS.probabilistic:
            probabilistic_predict(ARGS.file, feature_selection=ARGS.feature_selection)
        if ARGS.neural:
            bh_predict(ARGS.file)
        if ARGS.other:
            sw_predict(ARGS.file)
