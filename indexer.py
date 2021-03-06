# DO NOT MODIFY CLASS NAME
import pickle


class Indexer:
    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def __init__(self, config):
        self.inverted_idx = {}
        self.postingDict = {}
        self.config = config
        self.to_remove = []
        self.postingDocs={}  # docs["2343211"]=[(term , tf),(),....]

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def add_new_doc(self, document):
        """
        This function perform indexing process for a document object.
        Saved information is captures via two dictionaries ('inverted index' and 'posting')
        :param document: a document need to be indexed.
        :return: -
        """

        document_dictionary = document.term_doc_dictionary
        # Go over each term in the doc
        self.update_docs_dict(document.tweet_id,document_dictionary,document.doc_length)

        for term in document_dictionary.keys():
            try:
                # Update inverted index and posting
                if term not in self.inverted_idx.keys():
                    self.inverted_idx[term] = 1
                else:
                    self.inverted_idx[term] += 1

                self.update_post_dict(document.tweet_id, document.tweet_date, document_dictionary, term)

            except:
                print('problem with the following key {}'.format(term[0]))

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def load_index(self, fn):
        """
        Loads a pre-computed index (or indices) so we can answer queries.
        Input:
            fn - file name of pickled index.
        """
        #pkl object ->[inverted_idx,posting_data,posting_document]
        path = self.config.saveFilesWithoutStem +"\\"+ fn
        try:
            if self.config.toStem:
                path = self.config.saveFilesWithStem +"\\"+ fn
        except:
            pass
        file = open(path, 'rb')
        object_file = pickle.load(file)
        return object_file


    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def save_index(self, fn):
        """
        pkl object ->[inverted_idx,posting_data,posting_document]

        Saves a pre-computed index (or indices) so we can save our work.
        Input:
              fn - file name of pickled index.
        """
        #
        path =  self.config.saveFilesWithoutStem + "\\"+ fn+ ".pkl"
        try:
            if self.config.toStem:
                path = self.config.saveFilesWithStem + "\\"+fn+ ".pkl"
        except:
            pass
        dict_lst=[self.inverted_idx,self.postingDict,self.postingDocs]
        with open(path, 'wb') as f:
            pickle.dump(dict_lst, f,protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    # feel free to change the signature and/or implementation of this function 
    # or drop altogether.
    def _is_term_exist_post(self, term):
        """
        Checks if a term exist in the dictionary.
        """

        return term in self.postingDict

    def _is_term_exist_inv(self, term):
        """
        Checks if a term exist in the dictionary.
        """

        return term in self.inverted_idx

    # feel free to change the signature and/or implementation of this function 
    # or drop altogether.
    def get_term_posting_list(self, term):
        """
        Return the posting list from the index for a term.
        """
        return self.postingDict[term] if self._is_term_exist_post(term) else []

    def get_num_of_term_in_docs(self, term):
        """
        Return the number of show of term in all document.
        """

        return self.inverted_idx[term] if self._is_term_exist_inv(term) else 0

    def update_post_dict(self, tweet_id,tweet_date,term_dict,term):
          """
          update the post dict
          :param tweet_id: tweet ID int
          :param local_dict: dict hold the loction
          :param term_dict: dict hold frequency
          :param tweet_date:
          :return:
          """
          max_tf=max(term_dict.values())
          tf = term_dict[term] / max_tf
          if term not in self.postingDict:
              self.postingDict[term] = [[tweet_id, term_dict[term] , tf ,len(term_dict),tweet_date]] #[ tweetID,trem preq,tf,num uniqe terms in tweet,max_tf,date]
          else:
              self.postingDict[term].append([tweet_id, term_dict[term] , tf ,len(term_dict),tweet_date])

    def update_docs_dict(self,tweet_id,document_dictionary,doc_length):

        if(tweet_id in self.postingDocs):
            return
        local_tf_dict=[] #this dictionary hold "term" and the tf value in this specific doc
        max_freq = max(document_dictionary.values())
        for term in document_dictionary:
            local_tf_dict.append((term,document_dictionary[term] / max_freq,doc_length)) #(term, tf,doc_length)
        self.postingDocs[tweet_id]=local_tf_dict

    def get_docs_dict(self):
        return self.postingDocs




