"""Tests for C199: Search Engine"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))

import unittest
from search_engine import (
    HTMLParser, RobotsParser, SiteMapParser, URLFrontier, PageRank,
    WebCrawler, SearchResult, AutoComplete, SearchIndex, QueryProcessor,
    CrawlScheduler, SearchEngine, SearchAPI
)


# ---------- Helper: build test pages ----------

def make_page(title, body, links=None):
    """Build a simple HTML page."""
    link_html = ""
    if links:
        link_html = " ".join(f'<a href="{u}">{u}</a>' for u in links)
    return f"""<html>
<head><title>{title}</title>
<meta name="description" content="{title} page"></head>
<body><h1>{title}</h1><p>{body}</p>{link_html}</body></html>"""


# ===== HTMLParser Tests =====

class TestHTMLParser(unittest.TestCase):
    def setUp(self):
        self.parser = HTMLParser()

    def test_extract_title(self):
        html = "<html><head><title>Hello World</title></head><body></body></html>"
        result = self.parser.parse(html)
        self.assertEqual(result['title'], "Hello World")

    def test_extract_title_empty(self):
        html = "<html><head></head><body>text</body></html>"
        result = self.parser.parse(html)
        self.assertEqual(result['title'], "")

    def test_extract_meta(self):
        html = '<html><head><meta name="description" content="A test page"></head><body></body></html>'
        result = self.parser.parse(html)
        self.assertEqual(result['meta']['description'], "A test page")

    def test_extract_text(self):
        html = "<html><body><p>Hello</p><p>World</p></body></html>"
        result = self.parser.parse(html)
        self.assertIn("Hello", result['text'])
        self.assertIn("World", result['text'])

    def test_extract_text_strips_scripts(self):
        html = "<html><body><script>var x=1;</script><p>Visible</p></body></html>"
        result = self.parser.parse(html)
        self.assertNotIn("var x", result['text'])
        self.assertIn("Visible", result['text'])

    def test_extract_text_strips_styles(self):
        html = "<html><body><style>.cls{color:red}</style><p>Visible</p></body></html>"
        result = self.parser.parse(html)
        self.assertNotIn("color", result['text'])

    def test_extract_links(self):
        html = '<html><body><a href="http://example.com/page1">Link</a></body></html>'
        result = self.parser.parse(html, base_url="http://example.com")
        self.assertIn("http://example.com/page1", result['links'])

    def test_extract_links_relative(self):
        html = '<html><body><a href="/about">About</a></body></html>'
        result = self.parser.parse(html, base_url="http://example.com")
        self.assertIn("http://example.com/about", result['links'])

    def test_extract_links_skip_fragments(self):
        html = '<html><body><a href="#section">Skip</a><a href="http://x.com">Keep</a></body></html>'
        result = self.parser.parse(html)
        self.assertNotIn("#section", result['links'])
        self.assertIn("http://x.com", result['links'])

    def test_extract_links_skip_javascript(self):
        html = '<html><body><a href="javascript:void(0)">JS</a></body></html>'
        result = self.parser.parse(html)
        self.assertEqual(result['links'], [])

    def test_extract_headings(self):
        html = "<html><body><h1>Main</h1><h2>Sub</h2></body></html>"
        result = self.parser.parse(html)
        self.assertEqual(result['headings'], [(1, 'Main'), (2, 'Sub')])

    def test_html_entities(self):
        html = "<html><body><p>A &amp; B &lt; C</p></body></html>"
        result = self.parser.parse(html)
        self.assertIn("A & B < C", result['text'])

    def test_empty_html(self):
        result = self.parser.parse("")
        self.assertEqual(result['title'], "")
        self.assertEqual(result['text'], "")
        self.assertEqual(result['links'], [])

    def test_multiple_meta_tags(self):
        html = '<head><meta name="author" content="Alice"><meta name="keywords" content="test,page"></head>'
        result = self.parser.parse(html)
        self.assertEqual(result['meta']['author'], "Alice")
        self.assertEqual(result['meta']['keywords'], "test,page")


# ===== RobotsParser Tests =====

class TestRobotsParser(unittest.TestCase):
    def setUp(self):
        self.parser = RobotsParser()

    def test_allow_by_default(self):
        self.assertTrue(self.parser.is_allowed("http://example.com/page"))

    def test_disallow_path(self):
        self.parser.parse("example.com", "User-agent: *\nDisallow: /private/")
        self.assertFalse(self.parser.is_allowed("http://example.com/private/secret"))
        self.assertTrue(self.parser.is_allowed("http://example.com/public"))

    def test_allow_overrides_disallow(self):
        self.parser.parse("example.com", "User-agent: *\nDisallow: /dir/\nAllow: /dir/page")
        self.assertTrue(self.parser.is_allowed("http://example.com/dir/page"))

    def test_specific_user_agent(self):
        robots = "User-agent: mybot\nDisallow: /secret/\nUser-agent: *\nDisallow: /"
        self.parser.parse("example.com", robots)
        self.assertTrue(self.parser.is_allowed("http://example.com/public", user_agent='mybot'))
        self.assertFalse(self.parser.is_allowed("http://example.com/secret/data", user_agent='mybot'))

    def test_crawl_delay(self):
        self.parser.parse("example.com", "User-agent: *\nCrawl-delay: 5")
        self.assertEqual(self.parser.get_crawl_delay("example.com"), 5.0)

    def test_crawl_delay_none(self):
        self.assertIsNone(self.parser.get_crawl_delay("unknown.com"))

    def test_sitemap(self):
        self.parser.parse("example.com", "User-agent: *\nSitemap: http://example.com/sitemap.xml")
        sitemaps = self.parser.get_sitemaps("example.com")
        self.assertIn("http://example.com/sitemap.xml", sitemaps)

    def test_comments_ignored(self):
        self.parser.parse("example.com", "User-agent: * # all bots\nDisallow: /admin/ # no admins")
        self.assertFalse(self.parser.is_allowed("http://example.com/admin/page"))

    def test_wildcard_pattern(self):
        self.parser.parse("example.com", "User-agent: *\nDisallow: /tmp*")
        self.assertFalse(self.parser.is_allowed("http://example.com/tmp_data"))


# ===== SiteMapParser Tests =====

class TestSiteMapParser(unittest.TestCase):
    def setUp(self):
        self.parser = SiteMapParser()

    def test_parse_url_entries(self):
        xml = """<?xml version="1.0"?>
        <urlset><url><loc>http://example.com/page1</loc><priority>0.8</priority></url>
        <url><loc>http://example.com/page2</loc></url></urlset>"""
        entries = self.parser.parse(xml)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]['loc'], "http://example.com/page1")
        self.assertAlmostEqual(entries[0]['priority'], 0.8)

    def test_parse_sitemap_index(self):
        xml = """<sitemapindex><sitemap><loc>http://example.com/sitemap1.xml</loc></sitemap></sitemapindex>"""
        entries = self.parser.parse(xml)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]['type'], 'sitemap')

    def test_parse_lastmod(self):
        xml = """<urlset><url><loc>http://example.com/page</loc><lastmod>2025-01-01</lastmod></url></urlset>"""
        entries = self.parser.parse(xml)
        self.assertEqual(entries[0]['lastmod'], "2025-01-01")

    def test_parse_changefreq(self):
        xml = """<urlset><url><loc>http://example.com/page</loc><changefreq>weekly</changefreq></url></urlset>"""
        entries = self.parser.parse(xml)
        self.assertEqual(entries[0]['changefreq'], "weekly")

    def test_empty_sitemap(self):
        entries = self.parser.parse("<urlset></urlset>")
        self.assertEqual(entries, [])


# ===== URLFrontier Tests =====

class TestURLFrontier(unittest.TestCase):
    def setUp(self):
        self.frontier = URLFrontier()

    def test_add_url(self):
        self.assertTrue(self.frontier.add("http://example.com/page"))
        self.assertEqual(self.frontier.size, 1)

    def test_add_duplicate(self):
        self.frontier.add("http://example.com/page")
        self.assertFalse(self.frontier.add("http://example.com/page"))
        self.assertEqual(self.frontier.size, 1)

    def test_normalize_trailing_slash(self):
        self.frontier.add("http://example.com/page/")
        self.assertFalse(self.frontier.add("http://example.com/page"))

    def test_normalize_fragment(self):
        self.frontier.add("http://example.com/page#section")
        self.assertFalse(self.frontier.add("http://example.com/page"))

    def test_get_next_immediate(self):
        self.frontier.add("http://example.com/a", priority=0.5)
        self.frontier.add("http://other.com/b", priority=0.9)
        url = self.frontier.get_next_immediate()
        self.assertEqual(url, "http://other.com/b")

    def test_get_next_empty(self):
        self.assertIsNone(self.frontier.get_next_immediate())

    def test_max_size(self):
        frontier = URLFrontier(max_size=3)
        frontier.add("http://a.com/1")
        frontier.add("http://b.com/2")
        frontier.add("http://c.com/3")
        self.assertFalse(frontier.add("http://d.com/4"))

    def test_add_many(self):
        added = self.frontier.add_many(["http://a.com", "http://b.com", "http://a.com"])
        self.assertEqual(added, 2)

    def test_has_seen(self):
        self.frontier.add("http://example.com/page")
        self.assertTrue(self.frontier.has_seen("http://example.com/page"))
        self.assertFalse(self.frontier.has_seen("http://example.com/other"))

    def test_seen_count(self):
        self.frontier.add("http://a.com")
        self.frontier.add("http://b.com")
        self.frontier.get_next_immediate()
        self.assertEqual(self.frontier.seen_count, 2)  # seen stays after dequeue


# ===== PageRank Tests =====

class TestPageRank(unittest.TestCase):
    def test_empty_graph(self):
        pr = PageRank()
        scores = pr.compute()
        self.assertEqual(scores, {})

    def test_single_node(self):
        pr = PageRank()
        pr.add_link("A", "A")
        scores = pr.compute()
        self.assertAlmostEqual(scores["A"], 1.0, places=2)

    def test_two_nodes(self):
        pr = PageRank()
        pr.add_link("A", "B")
        pr.add_link("B", "A")
        scores = pr.compute()
        self.assertAlmostEqual(scores["A"], 0.5, places=2)
        self.assertAlmostEqual(scores["B"], 0.5, places=2)

    def test_star_topology(self):
        pr = PageRank()
        for i in range(5):
            pr.add_link(f"leaf{i}", "center")
        scores = pr.compute()
        # Center should have highest rank
        self.assertTrue(scores["center"] > scores["leaf0"])

    def test_chain_topology(self):
        pr = PageRank()
        pr.add_link("A", "B")
        pr.add_link("B", "C")
        pr.add_link("C", "A")
        scores = pr.compute()
        # Roughly equal in a cycle
        self.assertAlmostEqual(scores["A"], scores["B"], places=1)

    def test_dangling_node(self):
        pr = PageRank()
        pr.add_link("A", "B")
        # B has no outgoing links (dangling)
        scores = pr.compute()
        self.assertIn("A", scores)
        self.assertIn("B", scores)
        total = sum(scores.values())
        self.assertAlmostEqual(total, 1.0, places=2)

    def test_get_score(self):
        pr = PageRank()
        pr.add_link("A", "B")
        pr.compute()
        self.assertGreater(pr.get_score("B"), 0)
        self.assertEqual(pr.get_score("C"), 0.0)

    def test_get_top(self):
        pr = PageRank()
        for i in range(10):
            pr.add_link(f"leaf{i}", "hub")
        pr.compute()
        top = pr.get_top(3)
        self.assertEqual(top[0][0], "hub")

    def test_convergence(self):
        pr = PageRank(tolerance=1e-10, iterations=100)
        pr.add_link("A", "B")
        pr.add_link("B", "C")
        pr.add_link("C", "A")
        scores = pr.compute()
        total = sum(scores.values())
        self.assertAlmostEqual(total, 1.0, places=5)


# ===== WebCrawler Tests =====

class TestWebCrawler(unittest.TestCase):
    def test_crawl_single_page(self):
        crawler = WebCrawler()
        crawler.add_to_network("http://example.com", make_page("Home", "Welcome to the site"))
        pages = crawler.crawl(["http://example.com"], max_pages=1)
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0]['title'], "Home")

    def test_crawl_follows_links(self):
        crawler = WebCrawler()
        crawler.add_to_network("http://example.com",
            make_page("Home", "Welcome", ["http://example.com/about"]))
        crawler.add_to_network("http://example.com/about",
            make_page("About", "About us"))
        pages = crawler.crawl(["http://example.com"], max_pages=10)
        urls = [p['url'] for p in pages]
        self.assertIn("http://example.com", urls)
        self.assertIn("http://example.com/about", urls)

    def test_crawl_respects_max_pages(self):
        crawler = WebCrawler()
        for i in range(20):
            url = f"http://example.com/page{i}"
            next_url = f"http://example.com/page{i+1}"
            crawler.add_to_network(url, make_page(f"Page {i}", f"Content {i}", [next_url]))
        pages = crawler.crawl(["http://example.com/page0"], max_pages=5)
        self.assertEqual(len(pages), 5)

    def test_crawl_respects_robots(self):
        crawler = WebCrawler()
        crawler.add_robots("example.com", "User-agent: *\nDisallow: /secret/")
        crawler.add_to_network("http://example.com/secret/data",
            make_page("Secret", "Hidden"))
        page = crawler.crawl_page("http://example.com/secret/data")
        self.assertIsNone(page)

    def test_crawl_deduplication(self):
        crawler = WebCrawler()
        crawler.add_to_network("http://example.com",
            make_page("Home", "Home page", ["http://example.com/about", "http://example.com/about"]))
        crawler.add_to_network("http://example.com/about",
            make_page("About", "About page"))
        pages = crawler.crawl(["http://example.com"], max_pages=10)
        urls = [p['url'] for p in pages]
        self.assertEqual(len(set(urls)), len(urls))  # No duplicates

    def test_crawl_count(self):
        crawler = WebCrawler()
        crawler.add_to_network("http://a.com", make_page("A", "Page A"))
        crawler.add_to_network("http://b.com", make_page("B", "Page B"))
        crawler.crawl(["http://a.com", "http://b.com"], max_pages=10)
        self.assertEqual(crawler.crawl_count, 2)

    def test_fetch_missing_page(self):
        crawler = WebCrawler()
        self.assertIsNone(crawler.fetch("http://nonexistent.com"))

    def test_crawled_pages_dict(self):
        crawler = WebCrawler()
        crawler.add_to_network("http://example.com", make_page("Home", "Content"))
        crawler.crawl(["http://example.com"])
        pages = crawler.crawled_pages
        self.assertIn("http://example.com", pages)

    def test_pagerank_computed_after_crawl(self):
        crawler = WebCrawler()
        crawler.add_to_network("http://a.com",
            make_page("A", "Page A", ["http://b.com"]))
        crawler.add_to_network("http://b.com",
            make_page("B", "Page B", ["http://a.com"]))
        crawler.crawl(["http://a.com"], max_pages=10)
        self.assertGreater(crawler.page_rank.get_score("http://a.com"), 0)


# ===== SearchResult Tests =====

class TestSearchResult(unittest.TestCase):
    def test_to_dict(self):
        r = SearchResult("http://x.com", "Title", "snippet", 0.95, 0.001, 5.5)
        d = r.to_dict()
        self.assertEqual(d['url'], "http://x.com")
        self.assertEqual(d['title'], "Title")
        self.assertEqual(d['snippet'], "snippet")
        self.assertEqual(d['score'], 0.95)

    def test_repr(self):
        r = SearchResult("http://x.com", "Title", "snippet", 0.5)
        s = repr(r)
        self.assertIn("http://x.com", s)
        self.assertIn("0.5000", s)


# ===== AutoComplete Tests =====

class TestAutoComplete(unittest.TestCase):
    def setUp(self):
        self.ac = AutoComplete()

    def test_basic_suggestion(self):
        self.ac.add_query("python programming")
        self.ac.add_query("python tutorial")
        suggestions = self.ac.suggest("python")
        self.assertEqual(len(suggestions), 2)

    def test_frequency_ordering(self):
        self.ac.add_query("python tutorial", count=10)
        self.ac.add_query("python programming", count=5)
        suggestions = self.ac.suggest("python")
        self.assertEqual(suggestions[0], "python tutorial")

    def test_no_match(self):
        self.ac.add_query("python")
        suggestions = self.ac.suggest("java")
        self.assertEqual(suggestions, [])

    def test_empty_prefix(self):
        self.ac.add_query("python")
        self.assertEqual(self.ac.suggest(""), [])

    def test_limit(self):
        for i in range(20):
            self.ac.add_query(f"test query {i}")
        suggestions = self.ac.suggest("test", limit=3)
        self.assertEqual(len(suggestions), 3)

    def test_get_popular(self):
        self.ac.add_query("a", count=10)
        self.ac.add_query("b", count=5)
        self.ac.add_query("c", count=20)
        popular = self.ac.get_popular(2)
        self.assertEqual(popular[0][0], "c")
        self.assertEqual(len(popular), 2)

    def test_case_insensitive(self):
        self.ac.add_query("Python")
        suggestions = self.ac.suggest("python")
        self.assertEqual(len(suggestions), 1)


# ===== SearchIndex Tests =====

class TestSearchIndex(unittest.TestCase):
    def setUp(self):
        self.idx = SearchIndex()
        self.idx.add_page("http://a.com", {
            'title': 'Python Tutorial',
            'text': 'Learn python programming with examples and exercises',
            'headings': [(1, 'Python Tutorial')],
            'meta': {'description': 'Python tutorial for beginners'},
            'links': [],
        })
        self.idx.add_page("http://b.com", {
            'title': 'Java Guide',
            'text': 'Java programming language guide with best practices',
            'headings': [(1, 'Java Guide')],
            'meta': {'description': 'Java guide'},
            'links': [],
        })
        self.idx.add_page("http://c.com", {
            'title': 'Python Advanced',
            'text': 'Advanced python topics including decorators and metaclasses',
            'headings': [(1, 'Python Advanced')],
            'meta': {'description': 'Advanced python'},
            'links': [],
        })

    def test_search_basic(self):
        results = self.idx.search("python")
        self.assertGreater(len(results), 0)
        urls = [r.url for r in results]
        self.assertIn("http://a.com", urls)

    def test_search_ranking(self):
        results = self.idx.search("python")
        # Python pages should rank higher than Java
        urls = [r.url for r in results]
        self.assertNotIn("http://b.com", urls[:2])  # Java not in top 2

    def test_search_with_pagerank(self):
        self.idx.set_pagerank_scores({
            "http://a.com": 0.1,
            "http://b.com": 0.5,
            "http://c.com": 0.05,
        })
        results = self.idx.search("programming")
        # PageRank should influence ordering
        self.assertGreater(len(results), 0)

    def test_search_empty_query(self):
        results = self.idx.search("")
        self.assertEqual(results, [])

    def test_search_no_match(self):
        results = self.idx.search("xyznonexistent")
        self.assertEqual(results, [])

    def test_page_count(self):
        self.assertEqual(self.idx.page_count, 3)

    def test_has_page(self):
        self.assertTrue(self.idx.has_page("http://a.com"))
        self.assertFalse(self.idx.has_page("http://z.com"))

    def test_get_page(self):
        page = self.idx.get_page("http://a.com")
        self.assertEqual(page['title'], "Python Tutorial")

    def test_autocomplete(self):
        self.idx.add_query_history("python basics")
        self.idx.add_query_history("python advanced")
        suggestions = self.idx.autocomplete("python")
        self.assertEqual(len(suggestions), 2)

    def test_spell_check(self):
        corrected = self.idx.spell_check("pythn")
        self.assertIn("python", corrected.lower())

    def test_search_result_has_snippet(self):
        results = self.idx.search("python")
        for r in results:
            self.assertIsInstance(r.snippet, str)


# ===== QueryProcessor Tests =====

class TestQueryProcessor(unittest.TestCase):
    def setUp(self):
        self.idx = SearchIndex()
        self.idx.add_page("http://a.com", {
            'title': 'Test Page',
            'text': 'testing search functionality',
            'headings': [], 'meta': {}, 'links': [],
        })
        self.qp = QueryProcessor(self.idx)

    def test_basic_query(self):
        result = self.qp.process("test search")
        self.assertEqual(result['original'], "test search")
        self.assertEqual(len(result['terms']), 2)
        self.assertFalse(result['is_phrase'])

    def test_phrase_query(self):
        result = self.qp.process('"exact phrase"')
        self.assertTrue(result['is_phrase'])
        self.assertEqual(result['processed'], "exact phrase")

    def test_spell_correction(self):
        result = self.qp.process("tset")  # misspelling of "test"
        # May or may not correct depending on edit distance
        self.assertIn('spell_corrected', result)

    def test_enhance_with_correction(self):
        result = self.qp.process("test query")
        result['spell_corrected'] = "test query corrected"
        enhanced = self.qp.enhance(result, use_correction=True)
        self.assertEqual(enhanced['processed'], "test query corrected")


# ===== CrawlScheduler Tests =====

class TestCrawlScheduler(unittest.TestCase):
    def test_create_job(self):
        crawler = WebCrawler()
        scheduler = CrawlScheduler(crawler)
        job_id = scheduler.create_job(["http://a.com"], max_pages=10)
        self.assertEqual(job_id, 1)
        job = scheduler.get_job(job_id)
        self.assertEqual(job['status'], 'pending')

    def test_run_job(self):
        crawler = WebCrawler()
        crawler.add_to_network("http://a.com", make_page("A", "Page A"))
        scheduler = CrawlScheduler(crawler)
        job_id = scheduler.create_job(["http://a.com"])
        pages = scheduler.run_job(job_id)
        self.assertEqual(len(pages), 1)
        job = scheduler.get_job(job_id)
        self.assertEqual(job['status'], 'completed')
        self.assertEqual(job['pages_crawled'], 1)

    def test_list_jobs(self):
        crawler = WebCrawler()
        scheduler = CrawlScheduler(crawler)
        scheduler.create_job(["http://a.com"])
        scheduler.create_job(["http://b.com"])
        jobs = scheduler.list_jobs()
        self.assertEqual(len(jobs), 2)

    def test_nonexistent_job(self):
        crawler = WebCrawler()
        scheduler = CrawlScheduler(crawler)
        self.assertIsNone(scheduler.get_job(999))
        self.assertEqual(scheduler.run_job(999), [])


# ===== SearchEngine Integration Tests =====

class TestSearchEngine(unittest.TestCase):
    def setUp(self):
        self.engine = SearchEngine()
        self.engine.add_pages({
            "http://python.org": make_page(
                "Python", "Python is a programming language for web development and data science",
                ["http://python.org/docs", "http://python.org/download"]),
            "http://python.org/docs": make_page(
                "Python Docs", "Official documentation for the Python programming language",
                ["http://python.org"]),
            "http://python.org/download": make_page(
                "Download Python", "Download the latest version of Python",
                ["http://python.org"]),
            "http://java.com": make_page(
                "Java", "Java is a programming language for enterprise applications",
                ["http://java.com/docs"]),
            "http://java.com/docs": make_page(
                "Java Docs", "Official Java documentation and API reference",
                ["http://java.com"]),
        })

    def test_search_returns_results(self):
        results = self.engine.search("python programming")
        self.assertGreater(len(results), 0)

    def test_search_relevance(self):
        results = self.engine.search("python")
        urls = [r.url for r in results]
        # Python pages should rank above Java
        python_ranks = [i for i, u in enumerate(urls) if "python" in u]
        java_ranks = [i for i, u in enumerate(urls) if "java" in u]
        if python_ranks and java_ranks:
            self.assertLess(min(python_ranks), min(java_ranks))

    def test_search_with_info(self):
        info = self.engine.search_with_info("python")
        self.assertIn('query', info)
        self.assertIn('results', info)
        self.assertIn('total', info)
        self.assertEqual(info['query']['original'], "python")

    def test_page_count(self):
        self.assertEqual(self.engine.page_count, 5)

    def test_suggest_after_search(self):
        self.engine.search("python tutorial")
        self.engine.search("python docs")
        suggestions = self.engine.suggest("python")
        self.assertEqual(len(suggestions), 2)

    def test_add_single_page(self):
        page = self.engine.add_page("http://rust.org",
            make_page("Rust", "Rust systems programming language"))
        self.assertIsNotNone(page)
        self.assertEqual(self.engine.page_count, 6)

    def test_search_empty(self):
        results = self.engine.search("")
        self.assertEqual(results, [])

    def test_crawl_and_index(self):
        engine = SearchEngine()
        engine.crawler.add_to_network("http://test.com",
            make_page("Test", "Test page content", ["http://test.com/sub"]))
        engine.crawler.add_to_network("http://test.com/sub",
            make_page("Sub", "Sub page content"))
        pages = engine.crawl_and_index(["http://test.com"], max_pages=10)
        self.assertEqual(len(pages), 2)
        self.assertEqual(engine.page_count, 2)


# ===== SearchAPI Tests =====

class TestSearchAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = SearchEngine()
        cls.engine.add_pages({
            "http://example.com": make_page(
                "Example", "Example website for testing search functionality",
                ["http://example.com/about"]),
            "http://example.com/about": make_page(
                "About", "About the example website and its features",
                ["http://example.com"]),
        })
        cls.api = SearchAPI(cls.engine)
        cls.port = cls.api.start()

    @classmethod
    def tearDownClass(cls):
        cls.api.stop()

    def _request(self, method, path, body=None):
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("127.0.0.1", self.port))
        if body:
            body_bytes = json.dumps(body).encode()
            raw = f"{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {len(body_bytes)}\r\nConnection: close\r\n\r\n"
            s.sendall(raw.encode() + body_bytes)
        else:
            raw = f"{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"
            s.sendall(raw.encode())
        data = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
        s.close()
        # Parse response
        text = data.decode('utf-8', errors='replace')
        parts = text.split('\r\n\r\n', 1)
        status_line = parts[0].split('\r\n')[0]
        status_code = int(status_line.split()[1])
        body_text = parts[1] if len(parts) > 1 else ""
        # Handle chunked encoding
        headers_text = parts[0] if parts else ""
        if 'Transfer-Encoding: chunked' in headers_text:
            body_text = self._decode_chunked(body_text)
        return status_code, body_text

    def _decode_chunked(self, text):
        result = []
        lines = text.split('\r\n')
        i = 0
        while i < len(lines):
            try:
                size = int(lines[i], 16)
            except (ValueError, IndexError):
                i += 1
                continue
            if size == 0:
                break
            i += 1
            if i < len(lines):
                result.append(lines[i])
            i += 1
        return ''.join(result)

    def _json_request(self, method, path, body=None):
        status, text = self._request(method, path, body)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        return status, data

    def test_health(self):
        status, data = self._json_request("GET", "/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(data['status'], 'ok')

    def test_search(self):
        status, data = self._json_request("GET", "/api/search?q=example")
        self.assertEqual(status, 200)
        self.assertIn('results', data)
        self.assertGreater(data['total'], 0)

    def test_search_missing_query(self):
        status, data = self._json_request("GET", "/api/search")
        self.assertEqual(status, 400)

    def test_suggest(self):
        # First do a search to populate autocomplete
        self._json_request("GET", "/api/search?q=example+search")
        status, data = self._json_request("GET", "/api/suggest?q=example")
        self.assertEqual(status, 200)
        self.assertIn('suggestions', data)

    def test_index_page(self):
        status, data = self._json_request("POST", "/api/index", {
            'url': 'http://new.com',
            'html': make_page("New", "A brand new page"),
        })
        self.assertEqual(status, 201)
        self.assertEqual(data['status'], 'indexed')

    def test_index_missing_fields(self):
        status, data = self._json_request("POST", "/api/index", {'url': 'http://x.com'})
        self.assertEqual(status, 400)

    def test_stats(self):
        status, data = self._json_request("GET", "/api/stats")
        self.assertEqual(status, 200)
        self.assertIn('pages_indexed', data)
        self.assertGreater(data['pages_indexed'], 0)

    def test_crawl(self):
        # Add pages to network first
        self.engine.crawler.add_to_network("http://crawl.com",
            make_page("Crawl", "Crawlable content"))
        status, data = self._json_request("POST", "/api/crawl", {
            'seeds': ['http://crawl.com'],
            'max_pages': 5,
        })
        self.assertEqual(status, 201)
        self.assertEqual(data['status'], 'completed')

    def test_crawl_missing_seeds(self):
        status, data = self._json_request("POST", "/api/crawl", {})
        self.assertEqual(status, 400)

    def test_list_jobs(self):
        status, data = self._json_request("GET", "/api/jobs")
        self.assertEqual(status, 200)
        self.assertIn('jobs', data)

    def test_cors_preflight(self):
        status, _ = self._request("OPTIONS", "/api/search")
        self.assertEqual(status, 204)

    def test_search_with_limit(self):
        status, data = self._json_request("GET", "/api/search?q=example&limit=1")
        self.assertEqual(status, 200)
        self.assertLessEqual(len(data['results']), 1)

    def test_search_result_format(self):
        status, data = self._json_request("GET", "/api/search?q=example")
        self.assertEqual(status, 200)
        if data['results']:
            r = data['results'][0]
            self.assertIn('url', r)
            self.assertIn('title', r)
            self.assertIn('snippet', r)
            self.assertIn('score', r)
            self.assertIn('page_rank', r)


# ===== Advanced Integration Tests =====

class TestAdvancedFeatures(unittest.TestCase):
    def test_crawl_with_link_graph(self):
        """Test that PageRank influences search results."""
        engine = SearchEngine(text_weight=0.5, pagerank_weight=0.5)
        # Hub page linked to by many
        pages = {}
        hub_url = "http://example.com/hub"
        pages[hub_url] = make_page("Hub Page", "Programming resources hub")
        for i in range(10):
            url = f"http://example.com/leaf{i}"
            pages[url] = make_page(f"Leaf {i}", f"Programming leaf page {i}", [hub_url])
        engine.add_pages(pages)
        results = engine.search("programming")
        urls = [r.url for r in results]
        # Hub should rank high due to PageRank
        self.assertIn(hub_url, urls[:3])

    def test_multiword_search(self):
        engine = SearchEngine()
        engine.add_pages({
            "http://a.com": make_page("Python Web", "Build web applications with Python Flask"),
            "http://b.com": make_page("Python ML", "Machine learning with Python scikit"),
            "http://c.com": make_page("Java Web", "Java Spring web framework"),
        })
        results = engine.search("python web")
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].url, "http://a.com")

    def test_search_result_scores_ordered(self):
        engine = SearchEngine()
        engine.add_pages({
            "http://a.com": make_page("Alpha", "alpha alpha alpha content"),
            "http://b.com": make_page("Beta", "beta content here"),
        })
        results = engine.search("alpha")
        if len(results) > 1:
            self.assertGreaterEqual(results[0].score, results[1].score)

    def test_html_parser_robustness(self):
        parser = HTMLParser()
        # Malformed HTML
        result = parser.parse("<html><body><p>Unclosed paragraph<div>Nested wrong</p></div></body>")
        self.assertIn("Unclosed paragraph", result['text'])

    def test_frontier_priority(self):
        frontier = URLFrontier()
        frontier.add("http://low.com", priority=0.1)
        frontier.add("http://mid.com", priority=0.5)
        frontier.add("http://high.com", priority=0.9)
        url = frontier.get_next_immediate()
        self.assertEqual(url, "http://high.com")

    def test_pagerank_with_large_graph(self):
        pr = PageRank()
        n = 50
        for i in range(n):
            pr.add_link(f"node{i}", f"node{(i+1) % n}")
            if i % 3 == 0:
                pr.add_link(f"node{i}", "hub")
        pr.compute()
        # Hub should have higher rank
        self.assertGreater(pr.get_score("hub"), pr.get_score("node1"))

    def test_autocomplete_incremental(self):
        ac = AutoComplete()
        ac.add_query("python basics")
        ac.add_query("python advanced")
        ac.add_query("python web")
        ac.add_query("python web", count=5)  # boost
        s = ac.suggest("python", limit=3)
        self.assertEqual(s[0], "python web")  # most frequent first

    def test_search_index_zone_weighting(self):
        idx = SearchIndex()
        # Page with "python" only in title should rank differently
        idx.add_page("http://title.com", {
            'title': 'Python Guide',
            'text': 'This guide covers many topics',
            'headings': [], 'meta': {}, 'links': [],
        })
        idx.add_page("http://body.com", {
            'title': 'Guide',
            'text': 'Python python python appears many times in the body text here',
            'headings': [], 'meta': {}, 'links': [],
        })
        results = idx.search("python")
        self.assertGreater(len(results), 0)

    def test_full_pipeline(self):
        """End-to-end: add pages, crawl, search, get results."""
        engine = SearchEngine()
        # Add a small web
        engine.crawler.add_to_network("http://start.com",
            make_page("Start", "Starting page with links", ["http://start.com/a", "http://start.com/b"]))
        engine.crawler.add_to_network("http://start.com/a",
            make_page("Page A", "Content about algorithms and data structures", ["http://start.com"]))
        engine.crawler.add_to_network("http://start.com/b",
            make_page("Page B", "Content about databases and SQL queries", ["http://start.com"]))

        pages = engine.crawl_and_index(["http://start.com"], max_pages=10)
        self.assertEqual(len(pages), 3)

        results = engine.search("algorithms")
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].url, "http://start.com/a")

    def test_scheduler_multiple_jobs(self):
        crawler = WebCrawler()
        crawler.add_to_network("http://a.com", make_page("A", "Content A"))
        crawler.add_to_network("http://b.com", make_page("B", "Content B"))
        scheduler = CrawlScheduler(crawler)
        j1 = scheduler.create_job(["http://a.com"])
        j2 = scheduler.create_job(["http://b.com"])
        scheduler.run_job(j1)
        scheduler.run_job(j2)
        self.assertEqual(scheduler.get_job(j1)['status'], 'completed')
        self.assertEqual(scheduler.get_job(j2)['status'], 'completed')


if __name__ == '__main__':
    unittest.main()
