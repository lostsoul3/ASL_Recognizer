import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = ["" for word_id in range(0,len(test_set.get_all_Xlengths()))]
    try:
        probabilities = []
        guesses = ["" for word_id in range(0, len(test_set.get_all_Xlengths()))]
            # TODO implement the recognizer
            # return probabilities, guesses
        for word_id in range(0, len(test_set.get_all_sequences())):
            current_sequence = test_set.get_item_sequences(word_id)
            current_X,current_lengths = test_set.get_item_Xlengths(word_id)
            best_model = None
            best_word = None
            max_score = None
            p={}
            for word in models:
                model = models[word]
                try:
                    score = model.score(current_X,current_lengths)
                #probabilities[word_id][word]=score
                    p[word] = score
                except:
                    p[word] = 0
                if max_score==None or score > max_score:
                    max_score = score
                    best_word = word
            probabilities.append(p)
            guesses[word_id] = (best_word)
    except:
        pass
    return probabilities,guesses
