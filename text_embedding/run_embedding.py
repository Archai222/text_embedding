from text_embedding import documents
from text_embedding import cooc
from text_embedding.documents import evaluate
from text_embedding.documents import TASKMAP

if __name__ == '__main__':
    represent, prepare, invariant = cooc.DisC(1, 'mult')

    print(represent)
    print(prepare)
    evaluate('mr', represent, prepare=prepare, invariant=invariant, verbose=True,
             intercept='mr' in TASKMAP['pairwise task'])