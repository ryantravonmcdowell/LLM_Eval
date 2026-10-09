import pytest
from conftest import record

from bioeval.schema import SchemaError, question_from_dict
from bioeval.textnorm import jaccard, normalize, shingles, word_count


class TestNormalize:
    def test_unifies_primes_and_arrows(self):
        assert normalize("5′→3′") == "5' to 3'"
        assert normalize("5’ -> 3’") == "5' to 3'"

    def test_superscripts_survive_as_powers(self):
        # NFKC alone would turn q² into q2 and break "q^2" patterns
        assert normalize("q² = 0.16") == "q^2 = 0.16"

    def test_dashes_and_greek(self):
        assert normalize("p = 1 − q") == "p = 1 - q"
        assert normalize("β cells") == "beta cells"

    def test_keeps_line_breaks(self):
        assert normalize("1. A\n2. B") == "1. a\n2. b"


def test_word_count_handles_hyphens_and_apostrophes():
    assert word_count("Gram-negative cells don't stain") == 4


def test_near_duplicate_similarity():
    a = shingles("What is genetic drift in small populations?")
    b = shingles("What is genetic drift in small populations")
    assert jaccard(a, b) == 1.0
    assert jaccard(a, shingles("Explain how insulin lowers blood glucose")) == 0.0


class TestSchema:
    def test_valid_record_parses(self):
        q = question_from_dict(record())
        assert q.id == "t-001"
        assert [f.kind for f in q.key_facts] == ["fact", "reasoning"]
        assert q.constraints.max_words == 40
        assert "at most 40 words" in q.full_prompt()

    @pytest.mark.parametrize(
        ("overrides", "path"),
        [
            ({"id": ""}, "id"),
            ({"subject": "astrology"}, "subject"),
            ({"difficulty": "brutal"}, "difficulty"),
            ({"key_facts": []}, "key_facts"),
            ({"key_facts": [{"id": "x", "description": "d", "patterns": []}]},
             "key_facts[0].patterns"),
            ({"key_facts": [{"id": "x", "description": "d", "patterns": ["a"], "weight": 0}]},
             "key_facts[0].weight"),
            ({"key_facts": [{"id": "x", "description": "d", "patterns": ["a"], "kind": "vibe"}]},
             "key_facts[0].kind"),
            ({"constraints": {"format": "haiku"}}, "constraints.format"),
            ({"constraints": {"max_words": -3}}, "constraints.max_words"),
            ({"misconceptions": [{"id": "m", "description": "d", "patterns": ["x"]}]},
             "misconceptions[0].example"),
        ],
    )
    def test_invalid_records_name_the_field(self, overrides, path):
        with pytest.raises(SchemaError) as exc:
            question_from_dict(record(**overrides))
        assert exc.value.field_path == path

    def test_missing_field(self):
        r = record()
        del r["prompt"]
        with pytest.raises(SchemaError, match="prompt: missing"):
            question_from_dict(r)
