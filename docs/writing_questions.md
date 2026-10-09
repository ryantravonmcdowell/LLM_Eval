# Writing questions and rubrics

Every question in `src/bioeval/data/biology_v1.jsonl` is one JSON object per line. A question is
only useful if its rubric grades a correct answer as correct and a wrong answer as wrong, so a
rubric is written and tested like code: `bioeval validate` must pass before anything is graded.

## Anatomy of a question

Abbreviated from `gen-001` in the dataset:

```json
{
  "id": "gen-001",
  "subject": "genetics",
  "difficulty": "medium",
  "prompt": "In a population in Hardy-Weinberg equilibrium, 16% of individuals show a recessive phenotype. What are the two allele frequencies, and what fraction of the population are heterozygous carriers?",
  "reference_answer": "The recessive phenotype frequency is q² = 0.16, so q = 0.4 and p = 1 − q = 0.6. The heterozygote frequency is 2pq = 2 × 0.6 × 0.4 = 0.48, so 48% of individuals are carriers.",
  "key_facts": [
    {"id": "q", "kind": "fact", "description": "q = 0.4", "patterns": ["\\bq ?= ?0?\\.4\\b"]},
    {"id": "carriers", "kind": "fact", "description": "Carriers are 0.48 (48%)",
     "patterns": ["0?\\.48\\b", "\\b48 ?%"], "weight": 2.0},
    {"id": "q_squared", "kind": "reasoning", "description": "Recognises q^2 = 0.16",
     "patterns": ["q\\^?2 ?= ?0?\\.16"]}
  ],
  "misconceptions": [
    {"id": "q_is_phenotype", "description": "Treats the phenotype frequency as q",
     "patterns": ["\\bq ?= ?0?\\.16\\b"], "example": "q = 0.16, so p = 0.84."}
  ],
  "constraints": {"max_words": 100}
}
```

| Field | Rule |
|---|---|
| `subject` | One of the nine subjects in `bioeval.schema.SUBJECTS`. |
| `difficulty` | `easy`, `medium` or `hard`. |
| `key_facts` | What a correct answer must state. `kind: "fact"` for a claim, `kind: "reasoning"` for an intermediate step. `weight` defaults to 1. |
| `misconceptions` | Known wrong answers. Each needs an `example` sentence that its own patterns catch. |
| `constraints` | Optional `max_words`, `min_words`, and `format` (`numbered_list`, `bullets`, `single_sentence`). These are added to the prompt the model sees. |

## Writing good key facts

* **One checkable claim per fact.** "Synthesis runs 5' to 3'" is a fact; "explains replication"
  is not.
* **Weight what matters.** The final numeric answer to a calculation usually deserves `weight: 2`;
  a supporting detail can take `0.5`.
* **Mark steps as reasoning.** A missing step is reported as a `reasoning_gap`, a different
  problem from a missing conclusion, and the feedback should say which one happened.
* **Give alternatives, not one phrasing.** Patterns are alternatives: any one match counts. List
  the common correct wordings (`"anti-?parallel"`, `"opposite orientation"`).

## Writing misconceptions

Use the mistakes students and models actually make, such as confusing the anterior and posterior
pituitary, or saying a rightward Bohr shift *raises* affinity. Then:

* Write the `example` as the misconception would really appear in an answer.
* **Watch for negation.** If the reference says "not 3' to 5'", a pattern that matches
  `3' to 5'` will flag the reference itself. The audit catches this
  (`reference_has_misconception`), and the fix is to anchor the pattern to the claim, for example
  `(synthesi[sz]\w*|adds?)[^.]{0,40}3' ?to ?5'`.

## How text is matched

Before matching, answers and references are normalised (`bioeval.textnorm.normalize`):
lower-cased; `′ ’` become `'`; `→` and `->` become ` to `; `− – —` become `-`; `×` becomes `x`;
`²` becomes `^2`; `β` becomes `beta`. Write patterns against that normalised form, in lower case.
Matching is a regular-expression *search*, case-insensitive and multi-line.

## What `bioeval validate` checks

| Check | Severity | Catches |
|---|---|---|
| `json_parse`, `schema` | error | malformed lines, missing or mistyped fields (with the field path) |
| `duplicate_id`, `duplicate_item_id` | error | reused question or rubric-item ids |
| `invalid_regex` | error | a pattern that does not compile |
| `reference_missing_fact` | error | the reference answer fails its own key fact |
| `reference_has_misconception` | error | the reference answer trips its own misconception |
| `misconception_example_mismatch` | error | a misconception's example is not caught by its patterns |
| `reference_violates_constraints` | error | the reference breaks its own length or format rule |
| `near_duplicate_prompt` | warning | two prompts at least 80% similar (word 3-gram Jaccard) |
| `no_misconceptions` | warning | a question that cannot detect factual errors |

The test suite also asserts that **every reference answer scores exactly 1.0** and that **every
misconception example is flagged** when appended to its reference, so a rubric regression fails CI.

## Review checklist

1. Is the reference answer correct, and would a domain expert accept it as complete?
2. Does each key fact name one claim, with the important ones weighted?
3. Are there correct phrasings the patterns would miss? Add them as alternatives.
4. Does each misconception reflect a real, common error?
5. Does `bioeval validate` pass with no errors?
6. Run a real model and read its lowest-scoring answers: a correct answer graded wrong is a rubric
   bug, and it is fixed in the dataset, not explained away in the report.
