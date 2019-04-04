import os
import numpy as np
from snorkel import SnorkelSession
from snorkel.parser import TSVDocPreprocessor
from snorkel.parser.spacy_parser import Spacy
from snorkel.parser.rule_parser import RuleBasedParser
from snorkel.parser import CorpusParser
from snorkel.models import Document, Sentence
from snorkel.models import StableLabel, GoldLabel
from snorkel.models import candidate_subclass
from snorkel.candidates import Ngrams, CandidateExtractor, DocCandidate, WholeSentence
from snorkel.matchers import FirstQueryMatcher, SecondQueryMatcher, Matcher
from snorkel.db_helpers import reload_annotator_labels, reload_annotator_labels_next
from snorkel.annotations import load_gold_labels
from snorkel.annotations import LabelAnnotator
from snorkel.learning import GenerativeModel
from snorkel.learning.pytorch import LSTM
from snorkel.lf_helpers import test_LF
from snorkel.annotations import save_marginals

session = SnorkelSession()
n_docs = 364000

doc_preprocessor = TSVDocPreprocessor('../data/snorkel_query_all.tsv', max_docs=n_docs)
corpus_parser = CorpusParser(parser=RuleBasedParser())
corpus_parser.apply(doc_preprocessor, count=n_docs)
print("Documents:", session.query(Document).count())
print("Sentences:", session.query(Sentence).count())

docs = session.query(Document).order_by(Document.name).all()

train_docs = set()
dev_docs   = set()
test_docs  = set()
tmp_docs   = set()

doc_info = dict()
with open('../data/snorkel_label_all.tsv', 'r') as slf:
    for sll in slf:
        items = sll.split('\t')
        if len(items) == 3:
            label = '1'
            if items[1] == '0':
                label = '-1'
            doc_info[items[0]] = [label, items[2].strip()]

i = 0
for i, doc in enumerate(docs):
    #if len(doc.sentences) == 1:
    #for s in doc.sentences:
    #print(doc_info[doc.name])
    i = i + 1
    if i == 1:
        tmp_docs.add(doc)
    sens = doc.sentences[0].text.split('|')
    if len(sens) != 2:
        continue
    if doc_info[doc.name][1] == 'eval':
        dev_docs.add(doc)
    elif doc_info[doc.name][1] == 'test':
        test_docs.add(doc)
    elif doc_info[doc.name][1] == 'train':
        train_docs.add(doc)

print(len(dev_docs))
print(len(test_docs))
print(len(train_docs))

# Define a candidate subclass relying on the entire document
FullDocCandidate = candidate_subclass('FullDocCandidate', ['docCandidate'])

# Set up candidate extraction
fullDoc = DocCandidate()
defaultMatcher = Matcher()
cand_extractor = CandidateExtractor(FullDocCandidate, [fullDoc], [defaultMatcher])

# Get all the documents from the database
#docs = session.query(Document).order_by(Document.name).all()

# Extract candidates
for i, documents in enumerate([train_docs, dev_docs, test_docs]):
    cand_extractor.apply(documents, split=i)
    print("Number of candidates:", session.query(FullDocCandidate).filter(FullDocCandidate.split == i).count())

print("***********************")

#Qnas = candidate_subclass('Quora', ['question1', 'question2'])
#whole_sentence = WholeSentence()
#first_query_matcher = FirstQueryMatcher()
#second_query_matcher = SecondQueryMatcher()
#cand_extractor = CandidateExtractor(Qnas, [whole_sentence, whole_sentence], [first_query_matcher, second_query_matcher])

def load_local_labels(session, candidate_class, doc_info, annotator_name='gold'):
    #candidates = session.query(candidate_class).all()
    #for c in candidates:
        #print(c)
    #    context_stable_ids = '~~'.join(x.name for x in c)
    #    query = session.query(StableLabel).filter(
    #        StableLabel.context_stable_ids == context_stable_ids
    #    )
    #    query = query.filter(StableLabel.annotator_name == annotator_name)
    i = 0
    for key, value in doc_info.items():
        # We check if the label already exists, in case this cell was already executed
        context_stable_ids = key
        query = session.query(StableLabel).filter(StableLabel.context_stable_ids == context_stable_ids)
        query = query.filter(StableLabel.annotator_name == annotator_name)
        #print(query.all())
        i = i + 1
        if len(query.all()) == 0:
            session.add(StableLabel(
                context_stable_ids=context_stable_ids,
                annotator_name=annotator_name,
                value=value[0],
            ))
            #label = GoldLabel(candidate=candidate, key="gold", value=context_stable_ids)
            #session.add(label)
    
    print (i)
    # Commit session
    session.commit()

    # Reload annotator labels
    reload_annotator_labels_next(session, candidate_class, annotator_name, split=0, filter_label_split=False)
    reload_annotator_labels_next(session, candidate_class, annotator_name, split=1, filter_label_split=False)
    reload_annotator_labels_next(session, candidate_class, annotator_name, split=2, filter_label_split=False)

print("************************")
missed = load_local_labels(session, FullDocCandidate, doc_info, annotator_name='gold')

L_gold_dev = load_gold_labels(session, annotator_name='gold', split=0)
#print("gohere7")
#print(L_gold_dev)
#quit()

def LF_same_words_percentage(c):
    sens = c.docCandidate.sentences[0].text.split('|')
    if len(sens) != 2:
        return 0
    words1 = set(sens[0].split(' '))
    words2 = set(sens[1].split(' '))
    len_inter = len(words1.intersection(words2))
    percent_inter = len_inter * 2.0 / (len(words1) + len(words2))
    if percent_inter > 0.5:
            return 1
    return 0

def LF_same_length_percentage_min(c):
    sens = c.docCandidate.sentences[0].text.split('|')
    if len(sens) != 2:
        return 0
    words1 = sens[0].split(' ')
    words2 = sens[1].split(' ')
    len_inter = abs(len(words1) - len(words2))
    min_len = min(len(words1), len(words2))
    percent_inter = len_inter * 1.0 / (min_len + 0.000001)
    if percent_inter > 0.1:
            return 0
    return 1

def LF_same_length_percentage_max(c):
    sens = c.docCandidate.sentences[0].text.split('|')
    if len(sens) != 2:
        return 0
    words1 = sens[0].split(' ')
    words2 = sens[1].split(' ')
    len_inter = abs(len(words1) - len(words2))
    max_len = max(len(words1), len(words2))
    percent_inter = len_inter * 1.0 / (max_len + 0.000001)
    if percent_inter > 0.05:
            return 0
    return 1

LFs = [
    LF_same_words_percentage,
    LF_same_length_percentage_min,
    LF_same_length_percentage_max,
]

labeled1 = []
labeled2 = []
labeled3 = []
for c in session.query(FullDocCandidate).filter(FullDocCandidate.split == 0).all():
    if LF_same_words_percentage(c) != 0:
        labeled1.append(c)
    if LF_same_length_percentage_min(c) != 0:
        labeled2.append(c)
    if LF_same_length_percentage_max(c) != 0:
        labeled3.append(c)
print("Number labeled:", len(labeled1))
print("Number labeled:", len(labeled2))
print("Number labeled:", len(labeled3))
tp, fp, tn, fn = test_LF(session, LF_same_words_percentage, split=1, annotator_name='gold')
tp, fp, tn, fn = test_LF(session, LF_same_length_percentage_min, split=1, annotator_name='gold')
tp, fp, tn, fn = test_LF(session, LF_same_length_percentage_max, split=1, annotator_name='gold')

print("********************")

labeler = LabelAnnotator(lfs=LFs)
np.random.seed(1701)
L_train = labeler.apply(split=0)
#print(L_train)
L_train = labeler.load_matrix(session, split=0)
#print(L_train)
r = L_train.get_candidate(session, 0)
print(r)
r = L_train.get_key(session, 0)
print(r)
r = L_train.lf_stats(session)
print(r)
#print(L_train)

print("***********************")
gen_model = GenerativeModel()
gen_model.train(L_train, epochs=100, decay=0.95, step_size=0.1 / L_train.shape[0], reg_param=1e-6)

print(gen_model.weights.lf_accuracy)

train_marginals = gen_model.marginals(L_train)

#print(train_marginals)

gen_model.learned_lf_stats()
#L_dev = labeler.apply_existing(split=1)
#tp, fp, tn, fn = gen_model.error_analysis(session, L_dev, L_gold_dev)
save_marginals(session, L_train, train_marginals)

print("*********************")

train_cands = session.query(FullDocCandidate).filter(FullDocCandidate.split == 0).order_by(FullDocCandidate.id).all()
dev_cands   = session.query(FullDocCandidate).filter(FullDocCandidate.split == 1).order_by(FullDocCandidate.id).all()
test_cands  = session.query(FullDocCandidate).filter(FullDocCandidate.split == 2).order_by(FullDocCandidate.id).all()
L_gold_dev  = load_gold_labels(session, annotator_name='gold', split=1)
L_gold_test = load_gold_labels(session, annotator_name='gold', split=2)

train_kwargs = {
    'lr':            0.01,
    'embedding_dim': 50,
    'hidden_dim':    50,
    'n_epochs':      10,
    'dropout':       0.25,
    'seed':          1701
}

lstm = LSTM(n_threads=None)
lstm.train(train_cands, train_marginals, X_dev=dev_cands, Y_dev=L_gold_dev, **train_kwargs)
p, r, f1 = lstm.score(test_cands, L_gold_test)
print(p)
print(r)
print(f1)
