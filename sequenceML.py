import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
def sequenceML(fasta_file, expression_file, test_size,\
                             random_state,\
                                     filename,\
                                         tokenizer):
    """summary_line
    a automated framework for the sequence to expression
    machine learning. it takes the fasta sequences, expression
    file from the expression analysis and then writes the pickle
    file for the automated machine learning. It goes through all
    the classifier and gives you all the classifier and you can select
    the one based on the precision and accuracy call. It writes pickle
    file with the highest protocol
    Keyword arguments:
    argument -- description
    fasta_file_: coming from the transcriptome assembly 
    test_size_ : define the test size for the machine split learning
    random_state_ : define the random state for the machine learning
    filename_ : for writing the machine learning classifier and the pickle files
    tokenizer_ size of the token, while optimizing this i found the token size of 
    fasta sequences should not be more than 4-6 tokens if the length of the sequences
    is not long. 
    Return: return_description
    """
    
    testsize = test_size
    random = random_state
    token = int(tokenizer)
    sequence_file_train_read = list(filter(None,[x.strip() for x in open(fasta_file).readlines()]))
    sequence_train_dict = {}
    for i in sequence_file_train_read:
        if i.startswith(">"):
            genome_path = i.strip()
            if i not in sequence_train_dict:
                sequence_train_dict[i] = ""
                continue
        sequence_train_dict[genome_path] += i.strip()
    ids = list(map(lambda n: n.replace(">",""),sequence_train_dict.keys()))
    sequences = list(sequence_train_dict.values())
    sequence_dataframe = pd.DataFrame([(i,j)for i,j in zip(ids, sequences)]). \
                                          rename(columns = {0: "ids", 1: "sequence"})
    sequence_dataframe["expression"] = pd.read_csv(expression_file)
    sequence_dataframe["class"] = sequence_dataframe["expression"].apply(lambda n: "1" if n == 0.1 else "2" \
                                                            if n == 0.2  else "3" if n == 0.4 else "4" if n == 0.6 \
                                                                                        else "5" if n == 0.8 else "6" if n == 0.3 \
                                                                                                        else "7" if n == 0.5 else n)
    def segement(x):
        return [x[i:i+token] for i in range(len(x)-token+1)]
    sequence_dataframe["segmentation"] = sequence_dataframe["sequence"].apply(lambda n: segement(n))
    store_segmentation = sequence_dataframe["segmentation"].to_list()
    for i in range(len(store_segmentation)):
        store_segmentation[i] = ' '.join(store_segmentation[i])
    store_segmentation_length = len(store_segmentation)
    segmentation_class = sequence_dataframe["class"].values
    vectorise_start = [3, 4, 5, 6]
    vectorise_stop = [3, 4, 5, 6]
    storing_count_vectorise = [CountVectorizer(ngram_range=(i,j)) for i,j in \
                                                 zip(vectorise_start, vectorise_stop)]
    storing_count_vectorise_multiple_iterator = [(vectorise_start[i], vectorise_stop[j+1]) \
                                                 for i in range(len(vectorise_start)-1) \
                                                             for j in range(len(vectorise_stop)-1)]
    multiple_iterators_optimization_store = [CountVectorizer(ngram_range=storing_count_vectorise_multiple_iterator[i]) 
                                                               for i in range(len(storing_count_vectorise_multiple_iterator))]
    storing_assignments = [f"optimize_model{i} = {j}" for 
                                      i,j in enumerate(multiple_iterators_optimization_store,0)]
    storing_model_transformation = [f"X{i} = {storing_assignments[i].split()[0]}.fit_transform(store_segmentation)" \
                                                                                    for i in range(len(storing_assignments))]
    storing_classifier = [f"X_train{i}, X_test{i}, y_train{i}, y_test{i} = train_test_split({i},segmentation_class, test_size={testsize}, random_state={random})" 
                                                                for i in range(len(([storing_model_transformation[i].split()[0] 
                                                                                                    for i in range(len(storing_model_transformation))])))]
    alpha_predictions = list(np.linspace(0.1,1,num = len(storing_classifier)))
    sequential_classifier = [f"sequential_classifier{i} = MultinomialNB({alpha_predictions[i]}).transform({storing_classifier[i].split()[0]} {storing_classifier[i].split()[2]})" \
                                                                                                            for i in range(len(alpha_predictions))]
    prediction_classifier = [f"pred{i} = sequential_classifier.predict({storing_classifier[i].split()[1]})" 
                                                                            for i in range(len(storing_classifier))]
    f = open(filename, "wb")
    pickle.dump(storing_classifier, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(storing_model_transformation, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(alpha_predictions, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(sequential_classifier, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(prediction_classifier, f, pickle.HIGHEST_PROTOCOL)
    file.close()
    return print("pickle file for the machine learning has been written")
