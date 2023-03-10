"""
Format of record analysis

review_id : likely unimportant. drop
user_id : maybe something, but likely to result in false positives? likely drop/ignore
business_id: maybe something. but unlikely. Certain restaurants may be more likely to
    rate less from their previous scores? likely drop, but not 100%
stars: output. to be dropped but kept for training.
useful, funny, cool : a large majority of these ratings are default not rated, as
    such, only records that have these values should be used for training.
    This will likely end up in poor accuracy for these, as it is very likely a record
    has no rating of these.

text: input. could be skimmed or altered to make input easier?
    words like: 'the', 'a' , 'my', 'it', etc.. are likely unimportant so may be removed
        tests with skim and no skim should be done
    punctuation is likely not important

date: to be removed/ignored

"""
from pandas import read_json, json_normalize

class DataPreprocessor:
    """
    Preprocessor for the yelp_academic_dataset_review.json dataset
    """

    def __init__(self):
        pass

    def process(self, data):
        """
            Preprocesses the given dataset:
                review_id, user_id, business_id     -> Dropped from the dataset. They carry no relevant information

        :param data:
        :return:
        """

        data = data.drop(labels="review_id", axis=1)
        data = data.drop(labels="user_id", axis=1)
        data = data.drop(labels="date", axis=1)
        # may not drop this label
        data = data.drop(labels="business_id", axis=1)

        return data

if __name__ == '__main__':
    d = read_json('yelp_academic_dataset_review.json', lines=True, chunksize=10000)
    print("data read")
    for data in d:
        preprocessor = DataPreprocessor()
        result = preprocessor.process(data)
        result.to_csv('out.csv', mode='a')
