from collections import Counter

from amlc.segment import Segmenter

VOCAB = Counter({"navi": 50, "mumbai": 400, "producer": 30, "technologies": 300, "champions": 20, "elite": 60,
                 "growth": 80, "morganica": 3, "mendez": 5, "summit": 40, "plutonian": 3, "a": 900, "an": 500,
                 "grow": 10, "th": 1})


def test_splits_concatenated_business_words():
    seg = Segmenter(VOCAB)
    assert seg("producernavimumbai") == "producer navi mumbai"
    assert seg("technologieschampions") == "technologies champions"
    assert seg("elitegrowth") == "elite growth"
    assert seg("morganicamendezsummitplutonian") == "morganica mendez summit plutonian"


def test_unknown_text_stays_whole_and_digits_are_separate():
    seg = Segmenter(VOCAB)
    assert seg("cxflh") == "cxflh"
    assert seg("elite24growth") == "elite 24 growth"
