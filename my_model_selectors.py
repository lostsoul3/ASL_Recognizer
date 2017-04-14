import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        min_bic_score = None
        min_model = None
        try:
            for n_components in range(self.min_n_components, self.max_n_components + 1):
                model = self.base_model(n_components)
                logL = model.score(self.X, self.lengths)
                bic_score = -2 * logL + (n_components*(n_components-1) + 2*len(self.X[0])*n_components)*np.log(len(self.X))
                if min_bic_score == None or min_bic_score > bic_score:
                    min_bic_score = bic_score
                    min_model = model
            return min_model

        except:
            return min_model



class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        maxDIC = None
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            logL = {}
            valid_words = []
            for trained in (self.words).keys():
                X_train, lengths_train = self.hwords[trained]
                try:
                    # find suitable model
                    model = GaussianHMM(n_components, covariance_type="diag", n_iter=1000,random_state=self.random_state, verbose=False).fit(X_train, lengths_train)
                    # calculate log probability for the model
                    logL[trained] = model.score(X_train, lengths_train)
                    valid_words.append(trained)
                except ValueError:
                    pass
            try:  # Calculate DIC = log(P(X(i)) - 1 / (M - 1)SUM(log(P(X(all but i))
                DIC = logL[self.this_word] - (1 / (len(valid_words) - 1)) * sum([logL[valid] for valid in valid_words if valid != self.this_word])  # find max DIC score and best model
                if maxDIC==None or DIC > maxDIC:
                    maxDIC = DIC
                    best_model = model
            except KeyError:
                    best_model = model
        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV

        kf_no = min(3, len(self.sequences))
        max_avg = None
        i_best = -1
        hmm_best = None
        if kf_no==1:
            for n in range(self.min_n_components, self.max_n_components + 1):
                model = self.base_model(n)
                try:
                    score = model.score(self.X,self.lengths)
                    if max_avg==None or score > max_avg:
                        hmm_best = model
                except:
                    pass
        else:
            split_method = KFold(kf_no)

            for n in range(self.min_n_components,self.max_n_components+1):
                try:
                    sum_logl = 0
                    model = None
                    length = 0
                    for cv_train_idx,cv_test_idx in split_method.split(self.sequences):
                        X_train,lengths_train = combine_sequences(cv_train_idx,self.sequences)
                        X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
                        model = GaussianHMM(n, covariance_type="diag", n_iter=1000,random_state=self.random_state, verbose=False).fit(X_train, lengths_train)
                        score = model.score(X_test, lengths_test)
                        sum_logl = sum_logl + score
                        length = length+1

                    avg_score = sum_logl/length
                    if max_avg==None or avg_score > max_avg:
                        max_avg = avg_score
                        hmm_best = model

                except:
                    pass

        return hmm_best