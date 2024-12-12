#!/usr/bin/env python3
# coding=utf-8

import sys
import codecs
import re
import string


### Indic pattern for numbers and punctuations.
triv_tokenizer_indic_pat=re.compile(r'(['+string.punctuation+r'\u0964\u0965'+r'])')
pat_num_seq=re.compile(r'([0-9]+ [,.:/] )+[0-9]+')


### Punctuations list
puncts = list(string.punctuation)

### Telugu acronyms
acronyms = ["రు","కి","మీ","మి","సెం","రా","సా","తె","మ","గం","ని","జూ","నం","నెం","రూ","డాక్టర్","ఎ","ఏ","బి","సి","డి","ఇ","ఎఫ్","జి","హెచ్","ఐ","జె","జే","కె","కే","ఎల్","ఎం","ఎమ్","ఎన్","ఒ","ఓ","పి","ఫి","క్యు","క్యూ","ఆర్","ఎస్","టి","తి","యు","ఉ","ఊ","వి","డబ్లు","డబ్లూ","ఎక్స్","ఏక్స్","వై","జెడ్","జేడ్","శ్రీమతి","శ్రి","డాక్టర్","ప్రొఫెసర్","డా","ఛి","చి","చిరంజీవి","సౌ","ల"]
EndPunctuation = re.compile(r"("+("|".join(acronyms))+")+\s*")


def tokenize(text):
    """tokenize string for Indian language scripts using Brahmi-derived scripts

    This tokenizer just tokenizes on the punctuation boundaries. This also includes punctuations for the Indian language scripts (the purna virama and the deergha virama). This is a language independent tokenizer

    Args:
        text (str): text to tokenize

    Returns:
        list: list of tokens

    """
    tok_str = triv_tokenizer_indic_pat.sub(r' \1 ',text.replace('\t',' '))

    s = re.sub(r'[ ]+',' ',tok_str).strip(' ')
   
    # do not tokenize numbers and dates
    new_s = ''
    prev = 0
    for m in pat_num_seq.finditer(s):
        start = m.start()
        end = m.end()
        if start>prev:
            new_s = new_s + s[prev:start]
            new_s = new_s + s[start:end].replace(' ','')
            prev = end
   
    new_s = new_s + s[prev:]
    s = new_s

    '''
    The following code is to handle the case of more than two dots. In such cases, each dot is considered as a new sentence. To avoid that if previous word is a dot and current word is a dot then sentence split should not happen. (assuming  consequtive dots are not the end of sentence). Also, combined the consequitive similar puntuations as single token
    '''
    tokens = re.split(r'[ ]',s)

    word_flag = True
    tokens_list = []
    token = ""
    for i in range(0, len(tokens)-1):
        if tokens[i]==tokens[i+1] and tokens[i] in puncts:
            token += tokens[i]
            word_flag = False
        else:
            word_flag = True
            token += tokens[i]

        if word_flag:
            tokens_list.append(token)
            token = ""

    token += tokens[-1]
    tokens_list.append(token)

    return tokens_list


def preprocess_data(data):
    """Preprocess the given text and return the processed text

    This preprocessing methods includes the following techniques:
    1) Replace all tab spaces with single space
    2) Replace the 0-width space with null character
    3) Seperate more than one dot(.) and '"' with single ' ' (example: ..." --> ... ")
    4) Seperate more than one dot(.) and "'" with single ' ' (example: ...' --> ... ')
    5) Seperate more than one dot(.) and '-' with ' ' (example: '...-' --> '... -')
    6) Multiple new lines replaced with single new line.
    7) Multiple carriage returns replaced with single '\r'
    8) Multiple white spaces replaced with single space
    9) Finally leading/trailing spaces are trimmed.

    Args:
        data (str): text to apply preprocessing techniques

    Returns:
        str: processed text

    """

    data = re.sub(r"[\t]+"," ", data) ## Tested
    data = re.sub(r'[\u200b\u200c\u200d]', r'', data) ## Tested
    data = re.sub(r'(\.+)("+)',r' \1 \2 ',data) ## Tested
    data = re.sub(r"(\.+)('+)",r" \1 \2 ",data) ## Tested
    data = re.sub(r'(\.+)(-+)',r' \1 \2 ',data) ## Tested

    
    try:
        temp = re.split("\n",data)
        for i in range(len(temp)):
            temp[i] = temp[i].strip()
            if(len(temp[i])>0):
                temp[i] += " "

    except Exception as e:
        print(e)
            
    data = "".join(temp)

    data = re.sub(r"[\n]+","\n",data)
    data = re.sub(r"[\r]+","\r",data)
    data = re.sub(r"[ ]+"," ",data)
    data = data.strip()
    return data



def sentence_tokenize(data, return_count=False):
    """Sentence tokenizer takes the text as input and return the list of sentences. This sentence_tokenize methods initially apply the modified indic word tokenizer and use the hand-crafted rules to split the given text into list of sentences

    Args:
        data (str): text to apply sentencification
        return_count (bool): Flag to return the number of sentences

    Returns:
        list: list of sentences

    """
    words = tokenize(data) ### List of tokens seperated by space
    sentences = []
    
    ### Sentence begin, end and sentence break flag.
    begin = 0
    end = 1
    break_sen = False
    
    ### Next word, previous word and previous word index.
    prev = 0
    next_word = ""
    prev_word = ""

    ### List of characters to omit as previous word
    exclude_prev_chars = [".", ", ", " "]

    ### List of end of sentence chars
    end_of_sen_chars = ['.', '\n', '\r', '?', '!']

    i = 1
    while(i<len(words)):

        ### Finiding out the previous word index
        # if words[i-1]!="." and words[i-1]!=" " and words[i-1]!=",":
        if words[i-1] not in exclude_prev_chars:
            prev = i-1
        else:
            prev = i-2

        ### Previous word and Present word from list of tokens.
        curr_word = words[i]
        if(prev>=0):
            prev_word = words[prev]

        '''
        This code snippet handle the single and double quotes. If the ending quotes is missing
        then the sentencification will be done in normal way. Otherwise, it will consider the 
        content within the quotes as single sentence.
        '''
        ### Checking for the quotes
        if '"' in prev_word:
            temp_index = i
            while(temp_index<len(words)):
                if('"' in words[temp_index]):
                    end = temp_index + 1
                    i = end
                    prev_word = words[i-1]
                    if(i<len(words)):
                        curr_word = words[i]
                    else:
                        curr_word = ""
                        break_sen = True
                    break
                temp_index += 1

        if "'" in prev_word:
            temp_index = i
            while(temp_index<len(words)):
                if("'" in words[temp_index]):
                    end = temp_index + 1
                    i = end
                    prev_word = words[i-1]
                    if(i<len(words)):
                        curr_word = words[i]
                    else:
                        curr_word = ""
                        break_sen = True
                    break
                temp_index += 1
        '''
        Quotation code ends here.
        '''


        ### Checking if the current word is a sentence break (only for dot symbols) then previous word should not be an acronym.
        flag = False
        for end_of_sen in end_of_sen_chars:
            if end_of_sen in curr_word:
                if end_of_sen==".":
                    if end_of_sen==curr_word:
                        flag=True
                        break
                else:
                    flag = True
                    break


        if(flag==True and i>0):
            temp = EndPunctuation.search(prev_word)
            match_word = ""
            if temp is not None:
                match_word = prev_word[temp.span()[0]:temp.span()[1]]
            if prev_word not in acronyms and match_word!=prev_word:
                end = i+1
                break_sen = True

        ### Breaking the sentence if the sentence break flag is set to TRUE.
        if break_sen:
            sent = " ".join(words[begin:end])
            sent = sent.replace("\n","")
            sent = sent.replace("\r","")
            sent = sent.strip()

            sentences.append(sent)
            begin = end
            break_sen = False


        ### index increment
        i += 1

    ### Remaining words (last sentence) considered as the last sentence.
    sent = " ".join(words[begin:])
    if(len(sent)>=1 and (sent!="\n" and sent!="\t" and sent!=" ")):
        sentences.append(sent)

    if(return_count):
        return sentences, len(sentences)
    else:
        return sentences



def word_tokenize(sent_list, return_count=False):
    """Word tokenizer takes the list of sentences as input and return the list of list of tokens as output

    This word_tokenize method initially apply the indic word tokenizer.

    Args:
        sent_list (list): list of sentences (output of sentence_tokenize)
        return_count (bool): Flag to return the number of tokens

    Returns:
        list: list of tokens

    """

    ### Tokens, token break, previous word index, next word
    tokens = []
    break_wrd = True
    prev = 0
    next_word = ""
    
    for i in range(len(sent_list)):

        ### Extracting each sentence in a sentence list
        sent = sent_list[i]

        ### Applied the modified indic tokenizer for the sentence
        words = tokenize(sent)

        ### Adding all tokens to the token list
        tokens.extend(words)

    if(return_count):
        return tokens, len(tokens)
    else:
        return tokens


### Function to remove punctuations from the token list
def remove_punctuation(tokens, return_count=False):
    """This method takes the list of tokens as input and return the list of cleaned tokens (punctuations will be replaced with null) as output.

    Args:
        tokens (list): list of tokens (output of word_tokenize)
        return_count (bool): Flag to return the number of tokens

    Returns:
        list: list of cleaned tokens

    """

    cleaned_tokens = []
    pattern = re.compile(r"["+"".join(puncts)+"]+")
    for i in range(len(tokens)):
        token = tokens[i]

        ### Replacing the punctuations in the word with null 
        token = pattern.sub(r'',token)

        token = token.strip()
        if token!="":
            cleaned_tokens.append(token)

    if(return_count):
        return cleaned_tokens, len(cleaned_tokens)
    else:
        return cleaned_tokens