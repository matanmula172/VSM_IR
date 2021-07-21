import sys
from dic_to_json import *


# calculate query tf_idf_value
def query_tf_idf(data, tokens):
    tf_dict = dict()

    for i in range(len(tokens)):
        tf_dict[tokens[i]] = (float(tokens.count(tokens[i])) / float(len(tokens)))
    return data.get_document_weight_arr(tf_dict)


# Given a list of ranked documents, write at most 50 of them to a text file
def create_ranked_query_docs(ranked_documents):
    with open("ranked_query_docs.txt", "w") as ranked_query_docs:
        for doc in ranked_documents[:50]:
            ranked_query_docs.write(doc[1] + '\n')


if __name__ == '__main__':
    # if command is create_index then create json file in the given path
    if sys.argv[1] == 'create_index':
        path = sys.argv[2]
        dump_json_file(path)
    else:
        # else init CorpusData object, query, index_path
        query = str(sys.argv[3:])
        index_path = str(sys.argv[2])
        data = CorpusData(filenames, index_path)
        # load inverted index
        inverted_index = dict(load_json_file(index_path))
        tf_idf_dict = inverted_index["tf_idf"]
        data.word_over_docs = inverted_index["words_over_docs"]
        data.num_documents = inverted_index["num_documents"]
        # tokenize and stem query
        tokens = stem_tokenized_text(remove_stop_words(tokenize_text_no_punctuation(query)))

        # calculate tf_idf for query
        tf_idf_query = query_tf_idf(data, tokens)

        # rank documents by cosine similarity
        ranked_documents = rank_documents(tf_idf_dict, tf_idf_query, list(data.word_over_docs.keys()))

        # save ranked_query_docs.txt file
        create_ranked_query_docs(ranked_documents)
