from gensim.models import word2vec
import gensim
import logging


def model_train(train_file_name, save_model_file):
    # Model training, generating word vectors
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus(train_file_name)
    model = gensim.models.Word2Vec(sentences, vector_size=200, epochs=20)  # Train the skip gram model; Default window=5
    model.save(save_model_file)


def make_words():
    def get_text(file):
        data = open(file, 'r', encoding='utf-8')
        entries = data.read().strip().split("\n")
        labels = []
        for en in entries:
            for line in en.splitlines():
                pieces = line.strip().split()
                if len(pieces) < 2:
                    continue
                labels.append(pieces[0])
        return labels

    train_text = get_text(file=r'CMeEE-V2_train_split.txt')
    valid_text = get_text(file=r'CMeEE-V2_dev_split.txt')
    test_text = get_text(file=r'CMeEE-V2_test_split.txt')
    all_text = train_text + valid_text + test_text
    result = ''
    for one in all_text:
        result += str(one + ' ')
    result_text = f'word.txt'
    file_handle = open(result_text, mode='w', encoding='utf-8')
    file_handle.write(result)
    file_handle.close()


if __name__ == '__main__':
    make_words()
    save_model_name = 'Word2Vec.network'
    model_train(r'word.txt', save_model_name)

    model_w2v = word2vec.Word2Vec.load(save_model_name)
    model_w2v.wv.save_word2vec_format("word_vce.txt", binary=False)

