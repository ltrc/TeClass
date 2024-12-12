import re
from TeluguTokenizer.tokenizer import *


def sentencify(data):
    '''
        This method is responsible to sentencify the given text and return the sentences joined by delimeter "\n"

        Input:
            @data: string
        Output:
            List of sentences
    '''
    cleaned_data = preprocess_data(data)
    sentences = sentence_tokenize(cleaned_data)
    return "\n".join(sentences)


def content_validation(article_content, sentencified_content):
    '''
        This function is responsile for checking whether text and sentencified text are same or not

        Input:
            @article_content -> article text
            @sentencified_content -> sentencified text
        Output:
            @status -> True/False
    '''
    
    ### Pattern to match all spaces, tabs and newlines
    pattern = re.compile(r"\s+")

    ### Replace all the matches with null
    article_content = preprocess_data(article_content)
    sentencified_content = preprocess_data(sentencified_content)

    article_content = re.sub(pattern, '', article_content)
    sentencified_content = re.sub(pattern, '', sentencified_content)

    # checking whether text-sentencified text are same or not
    if(article_content==sentencified_content):
        return True

    else:
        return False


### =================================== xxx =====================================
###                            Summary Criteria Checking
### =================================== xxx =====================================

### Generating Ngrams given a list of tokens and the ngram (integer)
# def generate_ngrams(tokens, ngram):
#     '''
#     This method is responsible to generate the ngrams given the list of tokens
#     Input:
#         @tokens -> List of tokens
#         @ngram  -> integer
#     Output:
#         @ngram_list -> List of ngrams
#     '''
#     ngram_list = []

#     for i in range(len(tokens)-ngram+1):
#         ngram_list.append(tuple(tokens[i:i+ngram]))

#     return ngram_list


### Compute the set of Extractive Fragments F(A, S) (the set of shared sequences of tokens in A and S)
def F(A, S, ng = 0):
    '''
        This method is responsible to compute the set of extractive fragments given the article and summary text

        Input:
            @A -> Article text
            @S -> Summary text
            @ng -> abstractivity type [0, 1, 2, 3]
        Output:
            Set of matched sequences.
    '''
    F = []
    i = j = 0
    i_hat = 0

    while i<len(S):

        f = []
        while j<len(A):

            if S[i]==A[j]:

                i_hat = i
                j_hat = j

                while i_hat<len(S) and j_hat<len(A) and S[i_hat] == A[j_hat]:
                    i_hat = i_hat + 1
                    j_hat = j_hat + 1

                if (i_hat-i) > ng and len(f)<(i_hat - i):
                    f = [S[n] for n in range(i, i_hat)]

                j = j_hat

            else:
              j = j + 1

        i = i + max(len(f),1)
        j = 0
        F.append(f)

    return set(tuple(e) for e in F)


### Compare abstractivity score
def get_abstractivity_score(article_content, summary_content, ng=0):
    '''
        This method is responsible to compute the abstractivity score using the Extractive Fragments F(A, S)

        Input:
            @article_content -> article text
            @summary_content -> summary text
        Output:
            @abstractivity -> abstractivity0 score
    '''
    ### Preprocess the article and summary with preprocess_data
    summary_content = preprocess_data(summary_content)
    article_content = preprocess_data(article_content)

    ### Tokenization of article and summary with sentence tokenize
    summary_sents = sentence_tokenize(summary_content)
    article_sents = sentence_tokenize(article_content)

    ### Tokenization of article and summary with word tokenize
    summary_tokens_with_punct = word_tokenize(summary_sents)
    article_tokens_with_punct = word_tokenize(article_sents)

    ### Remove the punctuations from the tokens list and within the tokens
    summary_tokens = remove_punctuation(summary_tokens_with_punct)
    article_tokens = remove_punctuation(article_tokens_with_punct)
    
    # ### Building ngrams (trigram) using article and summary tokens
    # article_trigrams = generate_ngrams(article_tokens, 3)
    # summary_trigrams = generate_ngrams(summary_tokens, 3)

    # Calculating Abstractivity-0 score for article-summary pair
    abstractivity = -1
    matched_fragments = F(article_tokens, summary_tokens, ng)
    sumF_abs = 0
    for f in matched_fragments:
      sumF_abs += len(f)
    total_stokens = len(summary_tokens)
    if(total_stokens>0):
        abstractivity = (round((100 - (sumF_abs/total_stokens)*100), 2))

    return abstractivity


# Find prefix pairs
def has_prefix(article_content, summary_content):
    '''
        This method is responsible to check if the article and summary contain the same prefix upto first 15 characters

        Input:
            @article_content -> article text
            @summary_content -> summary text
        Output:
            @status -> True/False
    '''
    ### Preprocess the article and summary with preprocess_data
    summary_content = preprocess_data(summary_content)
    article_content = preprocess_data(article_content)

    ### Tokenization of article and summary with sentence tokenize
    summary_sentences = sentence_tokenize(summary_content)
    article_sentences = sentence_tokenize(article_content)
    
    # remove "city_name: " from the front
    article_sentences[0] = re.sub(r'^[^a-zA-Z\d]{1,15}:\s', '', article_sentences[0])
    summary_sentences[0] = re.sub(r'^[^a-zA-Z\d]{1,15}:\s', '', summary_sentences[0])
    print("Article: ", article_sentences[0])
    print("Summary: ", summary_sentences[0])
    
    if len(article_sentences) >= len(summary_sentences):
        for i in range(len(summary_sentences)):
            if not summary_sentences[i] == article_sentences[i]:
                if not summary_sentences[i] == article_sentences[i][:-1]:
                    return False

    else:
        return False

    return True


### Calculatig compression score
def get_compression_score(article_content, summary_content):
    '''
        This method is responsible to compute the compression score given the article and summary text

        Input:
            @article_content -> article text
            @summary_content -> summary text
        Output:
            @compression -> compression score
    '''
    ### Preprocess the article and summary with preprocess_data
    summary_content = preprocess_data(summary_content)
    article_content = preprocess_data(article_content)

    ### Tokenization of article and summary with sentence tokenize
    summary_sentences = sentence_tokenize(summary_content)
    article_sentences = sentence_tokenize(article_content)
    
    ### Tokenization of article and summary with indicnlp trivial_tokenize
    summary_tokens = word_tokenize(summary_sentences)
    article_tokens = word_tokenize(article_sentences)

    # Calculating Compression for article-summary pair
    compression = round(100-(len(summary_tokens)/len(article_tokens))*100,2)

    return compression