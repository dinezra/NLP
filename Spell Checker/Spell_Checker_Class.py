import math
import random
import string
from collections import Counter
import numpy as np
from .utils import normalize_text

# suggestion_prob.apeend((alpha * sentence_prob))
#

class Spell_Checker:
    """The class implements a context sensitive spell checker. The corrections
        are done in the Noisy Channel framework, based on a language model and
        an error distribution model.
    """

    def __init__(self,  lm=None):
        """Initializing a spell checker object with a language model as an
        instance  variable.

        Args:
            lm: a language model object. Defaults to None.
        """
        self.lm = lm
        self.error_table = None

    def oov(self,word):

        if word in self.lm.vocabulary: return False
        return word

    def candidate_edit_dis_1(self,word,edit=1):
        latters = 'abcdefghijklmnopqrstuvwxyz'
        dict_candidate ={}
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        for left_word, right_word in splits:
            if right_word != '':
                del_word = left_word + right_word[1:]
                if left_word == '' :
                    left_word_ = '#'
                else:
                    left_word_ = left_word
                if not self.oov(del_word):
                    dict_candidate[(word, del_word, 'deletion', edit)] = left_word_[-1] + right_word[0]
            if len(right_word) > 1:
                transe_word = left_word + right_word[1] + right_word[0] + right_word[2:]
                if  not self.oov(transe_word) and transe_word != word:
                    dict_candidate[(word, transe_word, 'transposition', edit)] = right_word[1] + right_word[0]
            for c in latters:
                if right_word != '':
                    replace_word = left_word + c + right_word[1:]
                    if not self.oov(replace_word) and replace_word!= word:

                        dict_candidate[(word, replace_word, 'substitution', edit)] = c + right_word[0]

                insert_word = left_word + c + right_word
                if left_word == '':
                    left_word_ = '#'
                else:
                    left_word_ = left_word
                if not self.oov(insert_word):
                    dict_candidate[(word, insert_word, 'insertion', edit)] = left_word_[-1] + c

        return dict_candidate

    def candidate_edit_dis_2(self, edit_1):
        dict_2_candidate ={}
        for org_word, word_after_action, action, edit in edit_1.keys():
            dict_2_candidate.update(self.candidate_edit_dis_1(word_after_action, edit=2))
        edit_1.update(dict_2_candidate)
        return edit_1


    def get_all_cadidate(self,text):
        all_candidate = {}
        candidat_edit1 = self.candidate_edit_dis_1(text)
        candidat_edit2 = self.candidate_edit_dis_2(candidat_edit1)
        all_candidate.update(candidat_edit1)
        all_candidate.update(candidat_edit2)
        return all_candidate



    def add_language_model(self, lm):
        """Adds the specified language model as an instance variable.
            (Replaces an older LM dictionary if set)

            Args:
                lm: a Spell_Checker.Language_Model object
        """
        self.lm=lm
        # self.prob_mistak = self.prob_mistake_calc()


    def add_error_tables(self, error_tables):
        """ Adds the specified dictionary of error tables as an instance variable.
            (Replaces an older value dictionary if set)

            Args:
            error_tables (dict): a dictionary of error tables in the format
            of the provided confusion matrices:
            https://www.dropbox.com/s/ic40soda29emt4a/spelling_confusion_matrices.py?dl=0
        """
        self.error_table = error_tables

    def get_value_error_table(self, action_type,latters):
        """
        :param tuple2check:
        :param latters:
        :return:
        """
        value = self.error_table[action_type][latters]
        return value

    def evaluate_text(self, text):
        """ Returns the log-likelihood of the specified text given the language
            model in use. Smoothing should be applied on texts containing OOV words

           Args:
               text (str): Text to evaluate.

           Returns:
               Float. The float should reflect the (log) probability.
        """
        return self.lm.evaluate_text(text)

    def get_norm_mistake_probabilities(self,all_candidates):
        """
        :param all_candidates:
        :return:
        """
        candidate_probabilitys=[]
        len(all_candidates.keys())
        for tuple2check,latters_change in all_candidates.items():
            org_word, change_word, action_type, edit_number = tuple2check[0], tuple2check[1], tuple2check[2], \
                                                              tuple2check[3]

            try:
                error_value = self.get_value_error_table(action_type,
                                                         latters_change)
                if error_value == 0: continue
                mistake_prob = error_value / self.lm.spell_dict_char[tuple(latters_change)]
            except (ZeroDivisionError, KeyError):
                mistake_prob = 0
            candidate_probabilitys.append((change_word,mistake_prob))
        return candidate_probabilitys

    def calc_prob_word_candidate(self,word,sentence):
        prob_candidate=[]
        all_candidates = self.get_all_cadidate(word)
        mistake_probs = self.get_norm_mistake_probabilities(all_candidates)
        for suggest_word, prob_mistake in mistake_probs:
            sentence_replace_word = sentence.replace(word, suggest_word)
            prob2sentence = self.evaluate_text(sentence_replace_word)
            if prob_mistake == 0:
                prob_candidate.append((suggest_word, math.pow(10, prob2sentence) * prob_mistake, sentence_replace_word))
            else: prob_candidate.append((suggest_word, math.log(math.pow(10, prob2sentence) * prob_mistake), sentence_replace_word))


        return prob_candidate

    def fix_sentence(self,sentence,candidate_word = None):
        finall_prob_candidate = []
        words = sentence.split()
        if candidate_word:
            finall_prob_candidate = self.calc_prob_word_candidate(candidate_word,sentence)
        else:
            for i in range(len(words)):
                finall_prob_candidate += self.calc_prob_word_candidate(words[i],sentence)
        if not finall_prob_candidate:
            return []
        return max(finall_prob_candidate,
                   key=lambda tup:tup[1])


    def check_oos_words(self,words):
        candidate_word =''
        for word in words:
            if self.oov(word) != False:
                has_oov = True
                candidate_word = word
                break
            has_oov = False
        return has_oov,candidate_word

    def spell_check(self, text, alpha):

        sentences = text.split(".")
        result = []

        for sentence in sentences:
            if sentence:
                sentence = normalize_text(sentence)
                words = sentence.split()
                has_oov,candidate_word = self.check_oos_words(words)
                if has_oov:
                    best_suggestion = self.fix_sentence(sentence,
                                                        candidate_word)
                    if not best_suggestion:
                        best_suggestion = sentence

                    else: best_suggestion = best_suggestion[2]

                else:
                    # Choose the suggestion with the highest probability
                    original_sentnece_prob = ("",self.evaluate_text(sentence)*alpha,sentence)
                    change_word_prob = self.fix_sentence(sentence)
                    if not change_word_prob: best_suggestion=original_sentnece_prob
                    else: best_suggestion = max(original_sentnece_prob,change_word_prob,key=lambda tup: tup[1])[2]

                new_sentence = best_suggestion
                result.append(new_sentence+'.')
                has_oov=False
        return ' '.join(result)


        """ Returns the most probable fix for the specified text. Use a simple
            noisy channel model if the number of tokens in the specified text is
            smaller than the length (n) of the language model.

            Args:
                text (str): the text to spell check.
                alpha (float): the probability of keeping a lexical word as is.

            Return:
                A modified string (or a copy of the original if no corrections are made.)

            for each word in words : argmax(prob(mistak-dina) * prob(ngram with suggest word)
        """

        normalized_text = normalize_text(text)
        sentences= normalized_text.split('.')
        result=[]
        for sentence in sentences:
            wrong_word = [word for word in sentence.split() if self.oov(word) != True]
            if not wrong_word:
                new_sentence = self.fix_sentence(sentence)
            else :  new_sentence = self.fix_sentence(sentence)


            # else:

            # result.append(new_sentence)

        return ".".join(result)




    ####################################################################
              # Inner class                                     #
    ####################################################################

    class Language_Model:
        """The class implements a Markov Language Model that learns a model from a given text.
          It supports language generation and the evaluation of a given string.
          The class can be applied on both word level and character level.
        """

        def __init__(self, n=3, chars=False):
            """Initializing a language model object.
            Args:
                n (int): the length of the markov unit (the n of the n-gram). Defaults to 3.
                chars (bool): True iff the model consists of ngrams of characters rather than word tokens.
                              Defaults to False
            """
            self.n = n
            self.model_dict = None
            self.model_dict_2 = None
            self.spell_dict_char = None
            self.vocabulary = set()
            self.proba = None
            self.chars=chars
            # a dictionary of the form {ngram:count}, holding counts of all ngrams in the specified text.
            # NOTE: This dictionary format is inefficient and insufficient (why?), therefore  you can (even encouraged to)
            # use a better data structure.
            # However, you are requested to support this format for two reasons:
            # (1) It is very straight forward and force you to understand the logic behind LM, and
            # (2) It serves as the normal form for the LM so we can call get_model_dictionary() and peek into you model.

        def create_ngram_dict(self, normalize_sentence, minus_n_dict):
            if minus_n_dict:
                N = self.n - 1
            else:
                N = self.n

            if self.chars:
                N = 2
                update_start_end = []
                words = ''.join(normalize_sentence)
                for word in words.split():
                    update_start_end.append('#'+word)
                words = ''.join(update_start_end
                                )

            else:
                words = []
                for sentence in normalize_sentence:
                    split_sentence = sentence.split(' ')
                    split_sentence = [word for word in split_sentence if word != '']
                    if not split_sentence: continue
                    words += tuple(['<sos>'] * (N - 1) + split_sentence + ['<sos>'] * (N - 1))
                    for word in split_sentence:
                        self.vocabulary.add(word)

            ngrams = []
            for i in range(len(words) - (N - 1)):
                ngram = tuple(words[i:i + N])
                if ngram[1] != '#' : ngrams.append(ngram)

            ngram_dict = Counter(ngrams)
            return ngram_dict

        #

        def build_model(self, text, minus_n_dict=False):  # should be called build_model
            """populates the instance variable model_dict.

                Args:
                    text (str): the text to construct the model from.
            """
            # build n-gram dicts
            sentences = text.split('.')
            normalize_sentence = [normalize_text(text) for text in sentences]
            self.model_dict = self.create_ngram_dict(normalize_sentence,
                                                     minus_n_dict)
            self.model_dict_2 = self.create_ngram_dict(normalize_sentence,
                                                       minus_n_dict=True)
            self.chars = True
            self.spell_dict_char =  self.create_ngram_dict(normalize_sentence,
                                                           minus_n_dict)
        # ddsda
        def get_model_dictionary(self):
            """Returns the dictionary class object
            """
            return self.model_dict

        def get_model_window_size(self):
            """Returning the size of the context window (the n in "n-gram")
            """
            return self.n


        def generate(self, context=None, n=20):
            """Returns a string of the specified length, generated by applying the language model
            to the specified seed context. If no context is specified the context should be sampled
            from the models' contexts distribution. Generation should stop before the n'th word if the
            contexts are exhausted. If the length of the specified context exceeds (or equal to)
            the specified n, the method should return a prefix of length n of the specified context.

                Args:
                    context (str): a seed context to start the generated string from. Defaults to None
                    n (int): the length of the string to be generated.

                Return:
                    String. The generated text.

            """
            def choose_word():
                return random.choices(list(self.model_dict.keys()))[0][-1]

            if context != None:
                context_cut = context.split()[-(self.n - 1):]
                while len(context_cut) < self.n-1:
                    word2add = choose_word()
                    context_cut.append(word2add)

            else:
                context_cut = ['<sos>']*(self.n-2)
                word2add= choose_word()
                context_cut.append(word2add)
            generated_text = context
            while len(generated_text.split()) < n:
                candidate_words,probabilitis = [],[]

                for w in self.vocabulary:
                    temp_tuple = tuple(context_cut + [w])
                    try:
                        prob=self.evaluate_text(temp_tuple)

                    except:
                        continue
                    probabilitis.append(prob)
                    candidate_words.append(w)

                if len(probabilitis) == 0:
                    best_word = random.choices(list(self.model_dict.keys()))[0][0]

                else:
                    # convert to numpy array and apply softmax
                    probs_np = np.exp(probabilitis) / np.sum(np.exp(probabilitis))

                    # convert back to list
                    norm_probs = list(probs_np)

                    best_word = random.choices(candidate_words,weights=norm_probs)[0]
                generated_text += ' ' + best_word
                context_cut = context_cut[1:] + [best_word]


            return generated_text

        def evaluate_text(self, text, Laplace=False):
            """Returns the log-likelihood of the specified text to be a product of the model.
                Laplace smoothing should be applied if necessary.

                Args:
                    text (str): Text to evaluate.

                Returns:
                    Float. The float should reflect the (log) probability.
            """
            N = self.n
            prob = 1
            text = normalize_text(text)
            # split by ngram
            words = text.split()
            while len(words) < self.n:
                words = ['<sos>']  + words
            for i in range(len(words)-N+1):
                cuurent_ngram = tuple((words[i:i+N]))
                cuurent_ngram_less1 = tuple((words[i:i+N-1]))
                if cuurent_ngram in self.model_dict and cuurent_ngram_less1 in self.model_dict_2:
                    mone = self.model_dict[cuurent_ngram]
                    mehane = self.model_dict_2[cuurent_ngram_less1]
                    prob *= (mone / mehane)
                else:   prob *= self.smooth(cuurent_ngram)
            return math.log(prob,10)

        def smooth(self, ngram):
            """Returns the smoothed (Laplace) probability of the specified ngram.

                Args:
                    ngram (str): the ngram to have its probability smoothed

                Returns:
                    float. The smoothed probability.
            """
            mone = self.model_dict[ngram] + 1
            mehane = self.model_dict_2[ngram[0:self.n - 1]] + len(self.vocabulary)
            return mone / mehane




