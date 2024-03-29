
import fasttext.util

if __name__ == "__main__":
    fasttext.util.download_model('en', if_exists='ignore')
    ft = fasttext.load_model('cc.en.300.bin')
    # Reduce word vectors to size of 25.
    fasttext.util.reduce_model(ft, 25)
    ft.save_model('cc.en.25.bin')
