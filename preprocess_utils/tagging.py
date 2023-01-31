from multiprocessing import Pool
import spacy
from spacy.matcher import Matcher
from tqdm import tqdm
import nltk
import json
import string
import re
__all__ = ['create_matcher_patterns', 'ground']


# the lemma of it/them/mine/.. is -PRON-

blacklist = set(["-PRON-", "actually", "likely", "possibly", "want",
                 "make", "my", "someone", "sometimes_people", "sometimes", "would", "want_to",
                 "one", "something", "sometimes", "everybody", "somebody", "could", "could_be"
                 ])


nltk.download('stopwords', quiet=True)
nltk_stopwords = nltk.corpus.stopwords.words('english')

# CHUNK_SIZE = 1

CPNET_VOCAB = None
PATTERN_PATH = None
nlp = None
matcher = None


def load_cpnet_vocab(cpnet_vocab_path):
    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        cpnet_vocab = [l.strip() for l in fin]
    cpnet_vocab = [c.replace("_", " ") for c in cpnet_vocab]
    return cpnet_vocab

def lemmatize(nlp, concept):

    doc = nlp(concept.replace("_", " "))
    lcs = set()
    lcs.add("_".join([token.lemma_ for token in doc]))  # all lemma
    return lcs

def load_matcher(nlp, pattern_path):
    with open(pattern_path, "r", encoding="utf8") as fin:
        all_patterns = json.load(fin)
    matcher = Matcher(nlp.vocab)
    # print('get the matcher')
    for concept, pattern in tqdm(all_patterns.items()):
        matcher.add(concept, [pattern])
    return matcher

def get_concept_position(sents, answers,stems,num_processes):
    res = []
    with Pool(num_processes) as p:
        res = list(tqdm(p.imap(tag_qa_pair, zip(sents, answers,stems)), total=len(sents)))
    return res   

def tag_qa_pair(qa_pair):

    # global nlp, matcher
    
    sents,answers,stem = qa_pair
    sent_pair,stem_pair,ans_pair = [],[],[]
    for s in sents:
        pos_pair = tag_concepts_pos(s,nlp,matcher)
        sent_pair.append(pos_pair)
    for a in answers:
        pos_pair = tag_concepts_pos(a,nlp,matcher)
        ans_pair.append(pos_pair)
    stem_pair = tag_concepts_pos(stem,nlp,matcher)
    res = {
        'statements':sent_pair,
        'answers':ans_pair,
        'stem':stem_pair
    }
    return res
    
    
    
# def tag_concepts_pos(s,nlp,matcher):
#     s = s.lower()
#     doc = nlp(s)
#     matches = matcher(doc)
#     pair = set()
#     split_pair = set()
#     for match_id, start, end in matches: 
#         span = doc[start:end].text
#         pair.add((start,end,span))
#         if end-start>1:
#             word_list = re.split(' |_',span)
#             if len(word_list) != end-start: 
#                 print(start,end,span,word_list)
#                 return []
#             for i in range(end-start):
#                 split_pair.add((start+i,start+i+1,word_list[i]))
#     # print(len(pair),len(split_pair))
#     pair= pair-split_pair
#     # print(pair)
#     return list(pair)

def prune(size,word_list):
    if len(word_list) != size: return False
    for i in range(size):
        if word_list[i] in nltk_stopwords:
            return False
    return True


def tag_concepts_pos(s,nlp,matcher):
    s = s.lower()
    doc = nlp(s)
    matches = matcher(doc)
    pair = set()
    split_pair = set()
    for match_id, start, end in matches: 
        span = doc[start:end].text
        word_list = span.split()
        size = end- start
        if prune(size,word_list):
            pair.add((start,end,span))
            if size >1 :
                for i in range(end-start):
                    split_pair.add((start+i,start+i+1,word_list[i]))
    pair= pair-split_pair
    return list(pair)


def tag(statement_path, cpnet_vocab_path, pattern_path, output_path, num_processes=1, debug=False):
    global PATTERN_PATH, CPNET_VOCAB
    if PATTERN_PATH is None:
        PATTERN_PATH = pattern_path
        CPNET_VOCAB = load_cpnet_vocab(cpnet_vocab_path)
        
    global nlp, matcher
    if nlp is None or matcher is None:
        nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
        nlp.add_pipe('sentencizer')
        matcher = load_matcher(nlp, PATTERN_PATH)
        
    sents = []
    answers = []
    stems = []
    with open(statement_path, 'r') as fin:
        lines = [line for line in fin]

    if debug:
        lines = lines[0:3]
        print(len(lines))
    for line in lines:
        sent_line = []
        ans_line = []
        if line == "":
            continue
        j = json.loads(line)
        for statement in j["statements"]:
            sent_line.append(statement["statement"])

        for answer in j["question"]["choices"]:
            ans = answer['text']
            # ans = " ".join(answer['text'].split("_"))
            try:
                assert all([i != "_" for i in ans])
            except Exception:
                print(ans)
            ans_line.append(ans)
        sents.append(sent_line)
        answers.append(ans_line)
        stems.append(j['question']['stem'])

    res = get_concept_position(sents, answers,stems,num_processes)
    

    # check_path(output_path)
    with open(output_path, 'w') as fout:
        # change write file to json format
        fout.write(json.dumps(res))
        # for dic in res:
        #     fout.write(json.dumps(dic) + '\n')

    print(f'grounded concepts saved to {output_path}')
    print()

if __name__ == "__main__":
    # create_matcher_patterns("../data/cpnet/concept.txt", "./matcher_res.txt", True)
    # ground("../data/statement/dev.statement.jsonl", "../data/cpnet/concept.txt", "../data/cpnet/matcher_patterns.json", "./ground_res.jsonl", 10, True)
    statement_path = '../data/csqa/statement/test.statement.jsonl'
    cpnet_vocab_path = '../data/cpnet/concept.txt'
    pattern_path = '../data/cpnet/matcher_patterns.json'
    output_path = '../data/obqa/tagged/test.jsonl'
    num_processes = 1
    debug=True
    tag(statement_path, cpnet_vocab_path, pattern_path, output_path, num_processes, debug)