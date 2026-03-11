"""Tests for C197: Information Retrieval."""
import math
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from ir import (
    tokenize, stem, analyze, Document, Posting, InvertedIndex,
    QueryExpander, RelevanceFeedback, Evaluator,
    WildcardSearcher, SpellCorrector, SnippetGenerator, ZoneIndex,
    STOP_WORDS,
)


# ============================================================
# Tokenization & Stemming
# ============================================================

def test_tokenize_basic():
    assert tokenize("Hello World") == ["hello", "world"]

def test_tokenize_punctuation():
    assert tokenize("it's a test, really!") == ["it", "s", "a", "test", "really"]

def test_tokenize_numbers():
    assert tokenize("version 3 and 42") == ["version", "3", "and", "42"]

def test_tokenize_empty():
    assert tokenize("") == []

def test_tokenize_mixed_case():
    assert tokenize("CamelCase HTTPServer") == ["camelcase", "httpserver"]

def test_stem_plural():
    assert stem("cats") == "cat"
    assert stem("dogs") == "dog"

def test_stem_ies():
    assert stem("stories") == "stori"

def test_stem_sses():
    assert stem("dresses") == "dress"

def test_stem_ed():
    assert stem("walked") == "walk"

def test_stem_ing():
    assert stem("walking") == "walk"

def test_stem_short_word():
    assert stem("the") == "the"
    assert stem("is") == "is"

def test_stem_y_to_i():
    assert stem("happy") == "happi"

def test_analyze_full():
    result = analyze("The quick brown fox jumps over the lazy dog")
    assert "the" not in result  # stop word removed
    assert "quick" in result or "quick" in [stem(t) for t in ["quick"]]

def test_analyze_no_stemming():
    result = analyze("running dogs", use_stemming=False)
    assert "running" in result

def test_analyze_keep_stopwords():
    result = analyze("the cat is here", remove_stopwords=False)
    assert "the" in result or stem("the") in result

def test_stop_words_present():
    assert "the" in STOP_WORDS
    assert "and" in STOP_WORDS
    assert "python" not in STOP_WORDS


# ============================================================
# Document
# ============================================================

def test_document_creation():
    doc = Document("d1", "hello world")
    assert doc.doc_id == "d1"
    assert doc.text == "hello world"
    assert doc.fields == {}

def test_document_with_fields():
    doc = Document("d1", "hello", fields={"category": "greeting"})
    assert doc.fields["category"] == "greeting"

def test_document_repr():
    doc = Document("d1", "test")
    assert "d1" in repr(doc)


# ============================================================
# Posting
# ============================================================

def test_posting_creation():
    p = Posting("d1", [0, 5, 10])
    assert p.doc_id == "d1"
    assert p.term_freq == 3
    assert p.positions == [0, 5, 10]


# ============================================================
# Inverted Index - Basic
# ============================================================

def make_index():
    idx = InvertedIndex(use_stemming=False, remove_stopwords=False)
    idx.add_documents([
        Document("d1", "the cat sat on the mat"),
        Document("d2", "the dog sat on the log"),
        Document("d3", "the cat chased the dog"),
    ])
    return idx

def test_index_num_docs():
    idx = make_index()
    assert idx.num_docs == 3

def test_index_doc_freq():
    idx = make_index()
    assert idx.doc_freq("cat") == 2  # d1, d3
    assert idx.doc_freq("dog") == 2  # d2, d3
    assert idx.doc_freq("the") == 3

def test_index_term_freq():
    idx = make_index()
    assert idx.term_freq("the", "d1") == 2  # "the cat sat on the mat"
    assert idx.term_freq("cat", "d1") == 1
    assert idx.term_freq("cat", "d2") == 0

def test_index_doc_lengths():
    idx = make_index()
    assert idx.doc_lengths["d1"] == 6  # 6 tokens

def test_index_avg_dl():
    idx = make_index()
    expected = (6 + 6 + 5) / 3
    assert abs(idx.avg_dl - expected) < 0.01

def test_index_get_postings():
    idx = make_index()
    postings = idx.get_postings("cat")
    assert len(postings) == 2
    doc_ids = {p.doc_id for p in postings}
    assert doc_ids == {"d1", "d3"}


# ============================================================
# TF-IDF
# ============================================================

def test_idf():
    idx = make_index()
    # "cat" appears in 2 of 3 docs
    idf = idx.idf("cat")
    assert abs(idf - math.log(3/2)) < 0.01

def test_idf_all_docs():
    idx = make_index()
    # "the" in all 3 docs
    idf = idx.idf("the")
    assert abs(idf - math.log(3/3)) < 0.01  # log(1) = 0

def test_idf_missing_term():
    idx = make_index()
    assert idx.idf("zebra") == 0.0

def test_tfidf_score():
    idx = make_index()
    score = idx.tfidf_score("cat", "d1")
    assert score > 0

def test_tfidf_score_missing():
    idx = make_index()
    score = idx.tfidf_score("cat", "d2")
    assert score == 0.0

def test_tfidf_search():
    idx = make_index()
    results = idx.tfidf_search("cat")
    doc_ids = [r[0] for r in results]
    assert "d1" in doc_ids
    assert "d3" in doc_ids
    assert "d2" not in doc_ids

def test_tfidf_search_empty():
    idx = make_index()
    assert idx.tfidf_search("") == []

def test_tfidf_search_multi_term():
    idx = make_index()
    results = idx.tfidf_search("cat dog")
    # d3 has both cat and dog, should score highest
    assert results[0][0] == "d3"


# ============================================================
# BM25
# ============================================================

def test_bm25_score():
    idx = make_index()
    score = idx.bm25_score("cat", "d1")
    assert score > 0

def test_bm25_score_missing():
    idx = make_index()
    assert idx.bm25_score("zebra", "d1") == 0.0

def test_bm25_search():
    idx = make_index()
    results = idx.bm25_search("cat")
    doc_ids = [r[0] for r in results]
    assert "d1" in doc_ids
    assert "d3" in doc_ids

def test_bm25_search_ranking():
    idx = InvertedIndex(use_stemming=False, remove_stopwords=False)
    idx.add_documents([
        Document("d1", "cat cat cat"),  # high TF for cat
        Document("d2", "cat dog bird"),  # low TF for cat
    ])
    results = idx.bm25_search("cat")
    assert results[0][0] == "d1"  # higher TF should rank first

def test_bm25_search_empty():
    idx = make_index()
    assert idx.bm25_search("") == []

def test_bm25_search_top_k():
    idx = make_index()
    results = idx.bm25_search("the", top_k=2)
    assert len(results) <= 2

def test_bm25_multi_term():
    idx = make_index()
    results = idx.bm25_search("cat dog")
    assert results[0][0] == "d3"  # has both terms


# ============================================================
# Boolean Search
# ============================================================

def test_boolean_and():
    idx = make_index()
    result = idx.boolean_search("cat AND dog")
    assert result == {"d3"}

def test_boolean_or():
    idx = make_index()
    result = idx.boolean_search("cat OR log")
    assert "d1" in result and "d2" in result and "d3" in result

def test_boolean_not():
    idx = make_index()
    result = idx.boolean_search("NOT cat")
    assert "d2" in result
    assert "d1" not in result
    assert "d3" not in result

def test_boolean_combined():
    idx = make_index()
    result = idx.boolean_search("cat AND NOT dog")
    assert result == {"d1"}

def test_boolean_empty():
    idx = make_index()
    result = idx.boolean_search("")
    assert result == set()

def test_boolean_single_term():
    idx = make_index()
    result = idx.boolean_search("mat")
    assert result == {"d1"}


# ============================================================
# Phrase Search
# ============================================================

def test_phrase_single_word():
    idx = make_index()
    result = idx.phrase_search("cat")
    assert "d1" in result and "d3" in result

def test_phrase_two_words():
    idx = make_index()
    result = idx.phrase_search("cat sat")
    assert result == {"d1"}

def test_phrase_not_found():
    idx = make_index()
    result = idx.phrase_search("dog mat")
    assert result == set()

def test_phrase_empty():
    idx = make_index()
    result = idx.phrase_search("")
    assert result == set()

def test_phrase_all_terms_present_but_not_consecutive():
    idx = make_index()
    # "cat" and "dog" both in d3 but not consecutive ("cat chased the dog")
    result = idx.phrase_search("cat dog")
    assert "d3" not in result


# ============================================================
# Proximity Search
# ============================================================

def test_proximity_adjacent():
    idx = make_index()
    result = idx.proximity_search("cat", "sat", 1)
    assert "d1" in result

def test_proximity_within_range():
    idx = make_index()
    # "cat" at pos 1, "dog" at pos 4 in d3 ("the cat chased the dog")
    result = idx.proximity_search("cat", "dog", 3)
    assert "d3" in result

def test_proximity_out_of_range():
    idx = make_index()
    result = idx.proximity_search("cat", "dog", 1)
    # distance is 3 in d3, so max_distance=1 should not match
    assert "d3" not in result

def test_proximity_missing_term():
    idx = make_index()
    result = idx.proximity_search("cat", "zebra", 10)
    assert result == set()


# ============================================================
# Faceted Search
# ============================================================

def test_faceted_search():
    idx = InvertedIndex(use_stemming=False, remove_stopwords=False)
    idx.add_documents([
        Document("d1", "python programming", fields={"language": "python", "level": "beginner"}),
        Document("d2", "java programming", fields={"language": "java", "level": "beginner"}),
        Document("d3", "python advanced topics", fields={"language": "python", "level": "advanced"}),
    ])
    results = idx.faceted_search("programming", {"language": "python"})
    doc_ids = [r[0] for r in results]
    assert "d1" in doc_ids
    assert "d2" not in doc_ids

def test_faceted_no_query():
    idx = InvertedIndex(use_stemming=False, remove_stopwords=False)
    idx.add_documents([
        Document("d1", "hello", fields={"type": "greeting"}),
        Document("d2", "goodbye", fields={"type": "farewell"}),
    ])
    results = idx.faceted_search("", {"type": "greeting"})
    doc_ids = [r[0] for r in results]
    assert "d1" in doc_ids
    assert "d2" not in doc_ids

def test_faceted_multi_value():
    idx = InvertedIndex(use_stemming=False, remove_stopwords=False)
    idx.add_documents([
        Document("d1", "text", fields={"tags": ["python", "ml"]}),
        Document("d2", "text", fields={"tags": ["java"]}),
    ])
    results = idx.faceted_search("", {"tags": "python"})
    assert len(results) == 1 and results[0][0] == "d1"

def test_faceted_or_within_facet():
    idx = InvertedIndex(use_stemming=False, remove_stopwords=False)
    idx.add_documents([
        Document("d1", "text", fields={"color": "red"}),
        Document("d2", "text", fields={"color": "blue"}),
        Document("d3", "text", fields={"color": "green"}),
    ])
    results = idx.faceted_search("", {"color": ["red", "blue"]})
    doc_ids = {r[0] for r in results}
    assert doc_ids == {"d1", "d2"}


# ============================================================
# Term Vectors & Cosine Similarity
# ============================================================

def test_term_vector():
    idx = make_index()
    vec = idx.term_vector("d1")
    assert isinstance(vec, dict)
    assert len(vec) > 0

def test_cosine_self_similarity():
    idx = make_index()
    vec = idx.term_vector("d1")
    sim = idx.cosine_similarity(vec, vec)
    assert abs(sim - 1.0) < 0.001

def test_cosine_orthogonal():
    idx = make_index()
    vec1 = {"a": 1.0, "b": 1.0}
    vec2 = {"c": 1.0, "d": 1.0}
    assert idx.cosine_similarity(vec1, vec2) == 0.0

def test_similar_documents():
    idx = make_index()
    # d1 and d2 share structure ("the X sat on the Y")
    sims = idx.similar_documents("d1")
    assert len(sims) > 0


# ============================================================
# Query Expansion
# ============================================================

def test_query_expansion():
    idx = InvertedIndex(use_stemming=False, remove_stopwords=False)
    idx.add_documents([
        Document("d1", "machine learning algorithms neural networks"),
        Document("d2", "deep learning neural networks training"),
        Document("d3", "cooking recipes food preparation"),
    ])
    expander = QueryExpander(idx, top_docs=2, top_terms=2)
    expanded = expander.expand("learning")
    # Should add terms from top docs about learning
    assert len(expanded) > len("learning")

def test_query_expansion_no_results():
    idx = InvertedIndex(use_stemming=False, remove_stopwords=False)
    idx.add_document(Document("d1", "hello world"))
    expander = QueryExpander(idx)
    result = expander.expand("nonexistent")
    assert result == "nonexistent"


# ============================================================
# Relevance Feedback
# ============================================================

def test_relevance_feedback_reweight():
    idx = InvertedIndex(use_stemming=False, remove_stopwords=False)
    idx.add_documents([
        Document("d1", "information retrieval systems"),
        Document("d2", "information extraction methods"),
        Document("d3", "database management systems"),
    ])
    rf = RelevanceFeedback(idx)
    modified = rf.reweight("information", relevant_ids=["d1"], non_relevant_ids=["d2"])
    assert isinstance(modified, dict)
    assert len(modified) > 0

def test_relevance_feedback_search():
    idx = InvertedIndex(use_stemming=False, remove_stopwords=False)
    idx.add_documents([
        Document("d1", "information retrieval systems"),
        Document("d2", "information extraction methods"),
        Document("d3", "database management systems"),
    ])
    rf = RelevanceFeedback(idx)
    q_vec = rf.reweight("systems", relevant_ids=["d1"])
    results = rf.search_with_feedback(q_vec)
    assert len(results) > 0


# ============================================================
# Evaluator
# ============================================================

def test_precision():
    assert Evaluator.precision(["d1", "d2", "d3"], ["d1", "d3"]) == 2/3

def test_precision_empty():
    assert Evaluator.precision([], ["d1"]) == 0.0

def test_recall():
    assert Evaluator.recall(["d1", "d2"], ["d1", "d2", "d3"]) == 2/3

def test_recall_empty_relevant():
    assert Evaluator.recall(["d1"], []) == 0.0

def test_f1():
    f1 = Evaluator.f1(["d1", "d2", "d3"], ["d1", "d3"])
    p = 2/3
    r = 1.0
    expected = 2 * p * r / (p + r)
    assert abs(f1 - expected) < 0.001

def test_f1_zero():
    assert Evaluator.f1(["d1"], ["d2"]) == 0.0

def test_precision_at_k():
    assert Evaluator.precision_at_k(["d1", "d2", "d3", "d4"], ["d1", "d3"], 2) == 0.5

def test_average_precision():
    # d1 is relevant at rank 1, d3 is relevant at rank 3
    ap = Evaluator.average_precision(["d1", "d2", "d3"], ["d1", "d3"])
    # P@1 = 1/1 = 1, P@3 = 2/3, AP = (1 + 2/3)/2 = 5/6
    assert abs(ap - 5/6) < 0.001

def test_average_precision_empty():
    assert Evaluator.average_precision(["d1"], []) == 0.0

def test_map():
    results1 = ["d1", "d2", "d3"]
    relevant1 = ["d1", "d3"]
    results2 = ["d2", "d1"]
    relevant2 = ["d1"]
    m = Evaluator.mean_average_precision([results1, results2], [relevant1, relevant2])
    assert m > 0

def test_dcg():
    dcg = Evaluator.dcg(["d1", "d2", "d3"], ["d1", "d3"])
    # d1 at pos 0: 1/log2(2)=1, d3 at pos 2: 1/log2(4)=0.5
    expected = 1/math.log2(2) + 1/math.log2(4)
    assert abs(dcg - expected) < 0.001

def test_ndcg():
    ndcg = Evaluator.ndcg(["d1", "d2", "d3"], ["d1", "d3"])
    assert 0 < ndcg <= 1.0

def test_ndcg_perfect():
    ndcg = Evaluator.ndcg(["d1", "d2"], ["d1", "d2"])
    assert abs(ndcg - 1.0) < 0.001

def test_ndcg_empty():
    assert Evaluator.ndcg(["d1"], []) == 0.0

def test_reciprocal_rank():
    assert Evaluator.reciprocal_rank(["d2", "d1", "d3"], ["d1"]) == 0.5

def test_reciprocal_rank_first():
    assert Evaluator.reciprocal_rank(["d1", "d2"], ["d1"]) == 1.0

def test_reciprocal_rank_not_found():
    assert Evaluator.reciprocal_rank(["d1", "d2"], ["d3"]) == 0.0

def test_mrr():
    r1 = ["d2", "d1"]  # RR = 0.5
    r2 = ["d3", "d3"]  # RR = 0
    rel1 = ["d1"]
    rel2 = ["d1"]
    mrr = Evaluator.mean_reciprocal_rank([r1, r2], [rel1, rel2])
    assert abs(mrr - 0.25) < 0.001


# ============================================================
# Wildcard Search
# ============================================================

def test_wildcard_no_wildcard():
    idx = InvertedIndex(use_stemming=False, remove_stopwords=False)
    idx.add_document(Document("d1", "cat dog"))
    ws = WildcardSearcher(idx)
    result = ws.search("cat")
    assert result == {"d1"}

def test_wildcard_prefix():
    idx = InvertedIndex(use_stemming=False, remove_stopwords=False)
    idx.add_documents([
        Document("d1", "category catalog"),
        Document("d2", "dog dolphin"),
    ])
    ws = WildcardSearcher(idx)
    result = ws.search("cat*")
    assert "d1" in result
    assert "d2" not in result

def test_wildcard_suffix():
    idx = InvertedIndex(use_stemming=False, remove_stopwords=False)
    idx.add_documents([
        Document("d1", "running jumping"),
        Document("d2", "swim climb"),
    ])
    ws = WildcardSearcher(idx)
    result = ws.search("*ing")
    assert "d1" in result

def test_wildcard_middle():
    idx = InvertedIndex(use_stemming=False, remove_stopwords=False)
    idx.add_documents([
        Document("d1", "color colour"),
    ])
    ws = WildcardSearcher(idx)
    result = ws.search("col*r")
    assert "d1" in result

def test_wildcard_no_match():
    idx = InvertedIndex(use_stemming=False, remove_stopwords=False)
    idx.add_document(Document("d1", "hello world"))
    ws = WildcardSearcher(idx)
    result = ws.search("xyz*")
    assert result == set()


# ============================================================
# Spell Correction
# ============================================================

def test_spell_correct_exact():
    idx = InvertedIndex(use_stemming=False, remove_stopwords=False)
    idx.add_document(Document("d1", "information retrieval"))
    sc = SpellCorrector(idx)
    assert sc.correct("information") == "information"

def test_spell_correct_typo():
    idx = InvertedIndex(use_stemming=False, remove_stopwords=False)
    idx.add_document(Document("d1", "information retrieval"))
    sc = SpellCorrector(idx)
    result = sc.correct("informaton")  # missing 'i'
    assert result == "information"

def test_spell_correct_query():
    idx = InvertedIndex(use_stemming=False, remove_stopwords=False)
    idx.add_document(Document("d1", "information retrieval"))
    sc = SpellCorrector(idx)
    corrected = sc.correct_query("informaton retreval")
    assert "information" in corrected

def test_spell_edit_distance():
    assert SpellCorrector.edit_distance("kitten", "sitting") == 3
    assert SpellCorrector.edit_distance("", "abc") == 3
    assert SpellCorrector.edit_distance("abc", "abc") == 0


# ============================================================
# Snippet Generation
# ============================================================

def test_snippet_basic():
    idx = InvertedIndex(use_stemming=False, remove_stopwords=False)
    idx.add_document(Document("d1", "the quick brown fox jumps over the lazy dog"))
    sg = SnippetGenerator(idx, snippet_length=5)
    snippet = sg.generate("d1", "fox")
    assert "**fox**" in snippet

def test_snippet_missing_doc():
    idx = InvertedIndex(use_stemming=False, remove_stopwords=False)
    sg = SnippetGenerator(idx)
    assert sg.generate("missing", "query") == ''

def test_snippet_highlight_multiple():
    idx = InvertedIndex(use_stemming=False, remove_stopwords=False)
    idx.add_document(Document("d1", "cat and dog and cat"))
    sg = SnippetGenerator(idx, snippet_length=10)
    snippet = sg.generate("d1", "cat", highlight_start="[", highlight_end="]")
    assert "[cat]" in snippet


# ============================================================
# Zone Index
# ============================================================

def test_zone_basic():
    zi = ZoneIndex(zone_weights={"title": 0.6, "body": 0.4}, use_stemming=False, remove_stopwords=False)
    zi.add_document("d1", {"title": "python tutorial", "body": "learn python programming basics"})
    zi.add_document("d2", {"title": "java guide", "body": "python is mentioned here"})
    results = zi.search("python")
    # d1 should rank higher (python in title + body)
    assert results[0][0] == "d1"

def test_zone_title_boost():
    zi = ZoneIndex(zone_weights={"title": 0.9, "body": 0.1}, use_stemming=False, remove_stopwords=False)
    zi.add_document("d1", {"title": "algorithms", "body": "python machine learning python python python"})
    zi.add_document("d2", {"title": "python basics", "body": "algorithms"})
    results = zi.search("python")
    # d2 should rank higher due to title weight
    assert results[0][0] == "d2"

def test_zone_no_results():
    zi = ZoneIndex(use_stemming=False, remove_stopwords=False)
    zi.add_document("d1", {"title": "hello", "body": "world"})
    results = zi.search("nonexistent")
    assert results == []


# ============================================================
# Stemming Integration
# ============================================================

def test_stemmed_index():
    idx = InvertedIndex(use_stemming=True, remove_stopwords=True)
    idx.add_documents([
        Document("d1", "The cats are running quickly"),
        Document("d2", "A dog runs slowly"),
    ])
    # "running" and "runs" should match via stemming
    results = idx.bm25_search("running")
    doc_ids = [r[0] for r in results]
    assert "d1" in doc_ids or "d2" in doc_ids

def test_stopword_removal():
    idx = InvertedIndex(use_stemming=False, remove_stopwords=True)
    idx.add_document(Document("d1", "the cat is here"))
    assert idx.doc_freq("the") == 0
    assert idx.doc_freq("cat") == 1


# ============================================================
# Edge Cases
# ============================================================

def test_single_doc_index():
    idx = InvertedIndex(use_stemming=False, remove_stopwords=False)
    idx.add_document(Document("d1", "hello world"))
    assert idx.num_docs == 1
    results = idx.bm25_search("hello")
    assert len(results) == 1

def test_duplicate_terms():
    idx = InvertedIndex(use_stemming=False, remove_stopwords=False)
    idx.add_document(Document("d1", "cat cat cat cat"))
    assert idx.term_freq("cat", "d1") == 4

def test_very_long_document():
    idx = InvertedIndex(use_stemming=False, remove_stopwords=False)
    text = " ".join([f"word{i}" for i in range(1000)])
    idx.add_document(Document("d1", text))
    assert idx.doc_lengths["d1"] == 1000

def test_unicode_in_text():
    idx = InvertedIndex(use_stemming=False, remove_stopwords=False)
    idx.add_document(Document("d1", "hello world cafe resume"))
    results = idx.bm25_search("hello")
    assert len(results) == 1

def test_add_documents_batch():
    idx = InvertedIndex(use_stemming=False, remove_stopwords=False)
    docs = [Document(f"d{i}", f"document number {i}") for i in range(100)]
    idx.add_documents(docs)
    assert idx.num_docs == 100


# ============================================================
# BM25 Parameters
# ============================================================

def test_bm25_custom_params():
    idx = make_index()
    s1 = idx.bm25_score("cat", "d1", k1=0.5, b=0.3)
    s2 = idx.bm25_score("cat", "d1", k1=2.0, b=0.9)
    # Different params should give different scores
    assert s1 != s2


# ============================================================
# Integration Tests
# ============================================================

def test_full_search_pipeline():
    """End-to-end: index docs, search, evaluate."""
    idx = InvertedIndex(use_stemming=True, remove_stopwords=True)
    docs = [
        Document("d1", "Information retrieval is the activity of obtaining information resources"),
        Document("d2", "Search engines use information retrieval techniques"),
        Document("d3", "Machine learning helps improve search results"),
        Document("d4", "Natural language processing for text analysis"),
        Document("d5", "Database systems store and retrieve data efficiently"),
    ]
    idx.add_documents(docs)

    # BM25 search
    results = idx.bm25_search("information retrieval")
    retrieved = [r[0] for r in results]
    relevant = ["d1", "d2"]

    # Evaluate
    p = Evaluator.precision(retrieved[:3], relevant)
    r = Evaluator.recall(retrieved[:3], relevant)
    assert p > 0 and r > 0

def test_query_expansion_improves_recall():
    """Query expansion should find more relevant docs."""
    idx = InvertedIndex(use_stemming=False, remove_stopwords=False)
    idx.add_documents([
        Document("d1", "neural network deep learning training"),
        Document("d2", "neural network architecture design"),
        Document("d3", "backpropagation gradient descent training"),
        Document("d4", "cooking recipes kitchen food"),
    ])

    # Original search
    orig_results = idx.bm25_search("neural", top_k=10)
    orig_docs = {r[0] for r in orig_results}

    # Expanded search
    expander = QueryExpander(idx, top_docs=2, top_terms=3)
    expanded = expander.expand("neural")
    exp_results = idx.bm25_search(expanded, top_k=10)
    exp_docs = {r[0] for r in exp_results}

    # Expanded should find at least as many relevant docs
    assert len(exp_docs) >= len(orig_docs)

def test_boolean_then_rank():
    """Boolean filter then rank with BM25."""
    idx = InvertedIndex(use_stemming=False, remove_stopwords=False)
    idx.add_documents([
        Document("d1", "python machine learning algorithms"),
        Document("d2", "python web development django"),
        Document("d3", "java machine learning algorithms"),
    ])
    # Boolean: must have python
    bool_results = idx.boolean_search("python")
    assert "d1" in bool_results and "d2" in bool_results and "d3" not in bool_results

    # Then rank by relevance to "machine learning" among boolean results
    scores = []
    for doc_id in bool_results:
        score = idx.bm25_score("machine learning", doc_id)
        scores.append((doc_id, score))
    scores.sort(key=lambda x: -x[1])
    assert scores[0][0] == "d1"


# ============================================================
# Run all tests
# ============================================================

if __name__ == "__main__":
    test_functions = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for fn in test_functions:
        try:
            fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"FAIL: {fn.__name__}: {e}")
    print(f"\n{passed} passed, {failed} failed out of {passed + failed} tests")
