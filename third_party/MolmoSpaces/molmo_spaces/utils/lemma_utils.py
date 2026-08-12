import functools

from nltk.corpus.reader.wordnet import Synset

from molmo_spaces.utils.synset_utils import wn

PHYSICAL_ENTITY_SYNSET = wn.synset("physical_entity.n.01")


def normalize_expression(text: str) -> str:
    if ".n." in text:
        text = simple_lemma(text)

    return text.strip().lower().replace("_", " ").strip().strip(".;/,'\"\\")


def is_physical_entity(synset: Synset | str) -> bool:
    if isinstance(synset, str):
        synset = wn.synset(synset)
    return PHYSICAL_ENTITY_SYNSET in synset.lowest_common_hypernyms(PHYSICAL_ENTITY_SYNSET)


@functools.lru_cache(maxsize=1000)
def best_lemma_via_specificity(synset_str: str, enforce_physical_entity: bool = True) -> str:
    synset = wn.synset(synset_str)
    cur_synset_is_physical_entity = is_physical_entity(synset)
    min_num_synsets = 100000
    best_lemma = None
    for ln in synset.lemma_names():
        if cur_synset_is_physical_entity or enforce_physical_entity:
            num_synsets = len([s for s in wn.synsets(ln, pos=wn.NOUN) if is_physical_entity(s)])
        else:
            num_synsets = len(wn.synsets(ln, pos=wn.NOUN))
        if 0 < num_synsets < min_num_synsets:
            min_num_synsets = num_synsets
            best_lemma = ln

    if enforce_physical_entity and best_lemma is None:
        best_lemma = best_lemma_via_specificity(synset_str, enforce_physical_entity=False)

    assert best_lemma is not None, f"Failed to find lemma for {synset_str}"

    return best_lemma


def simple_lemma(synset_str: str) -> str:
    return wn.synset(synset_str).lemma_names()[0]
