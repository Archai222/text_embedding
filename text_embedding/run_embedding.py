from text_embedding import documents
from text_embedding import cooc
from text_embedding.documents import evaluate
from text_embedding.documents import TASKMAP
import os

if __name__ == '__main__':
    represent, prepare, invariant = cooc.DisC(1, 'mult')
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
    print(represent)
    print(prepare)
    # evaluate('cr', represent, prepare=prepare, invariant=invariant, verbose=True,
    #          intercept='imdb' in TASKMAP['train-test split'])

    evaluate('cr', represent, prepare=prepare, invariant=invariant, verbose=True)