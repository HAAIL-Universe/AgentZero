"""
C199: Search Engine
Composes C197 (Information Retrieval) + C016 (HTTP Server)

A complete web search engine with:
- WebCrawler: fetches and parses HTML pages (simulated network)
- HTMLParser: extracts text, links, title, meta from HTML
- PageRank: link-analysis ranking algorithm
- URLFrontier: priority-based crawl scheduling with politeness
- SearchIndex: unified index combining text relevance + PageRank
- QueryProcessor: query parsing, spelling correction, suggestions
- SearchAPI: REST API over HTTP for search, indexing, crawl management
- RobotsParser: robots.txt compliance
- SiteMap: sitemap.xml parsing
- CrawlScheduler: manages crawl jobs with rate limiting
- SearchResult: rich result objects with snippets and highlights
- AutoComplete: prefix-based query suggestion
"""

import sys, os, re, math, time, json, hashlib
from collections import defaultdict, deque
from urllib.parse import urlparse, urljoin, unquote, quote, parse_qs

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C197_information_retrieval'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C016_http_server'))

from ir import (
    Document, InvertedIndex, QueryExpander, SpellCorrector,
    SnippetGenerator, Evaluator, ZoneIndex, tokenize, analyze
)
from http_server import HTTPServer, Router, Request, Response


# ---------- HTML Parser ----------

class HTMLParser:
    """Extract text, links, title, and meta from HTML."""

    def __init__(self):
        pass

    def parse(self, html, base_url=""):
        """Parse HTML and return structured data."""
        result = {
            'title': self._extract_title(html),
            'meta': self._extract_meta(html),
            'text': self._extract_text(html),
            'links': self._extract_links(html, base_url),
            'headings': self._extract_headings(html),
        }
        return result

    def _extract_title(self, html):
        m = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else ""

    def _extract_meta(self, html):
        meta = {}
        for m in re.finditer(r'<meta\s+([^>]+?)/?>', html, re.IGNORECASE):
            attrs = self._parse_attrs(m.group(1))
            name = attrs.get('name', attrs.get('property', '')).lower()
            content = attrs.get('content', '')
            if name and content:
                meta[name] = content
        return meta

    def _parse_attrs(self, attr_str):
        attrs = {}
        for m in re.finditer(r'(\w[\w-]*)=["\']([^"\']*)["\']', attr_str):
            attrs[m.group(1).lower()] = m.group(2)
        return attrs

    def _extract_text(self, html):
        # Remove script and style
        text = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html, flags=re.IGNORECASE | re.DOTALL)
        # Remove tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Decode entities
        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        text = text.replace('&quot;', '"').replace('&#39;', "'").replace('&nbsp;', ' ')
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _extract_links(self, html, base_url):
        links = []
        for m in re.finditer(r'<a\s+([^>]*?)>', html, re.IGNORECASE):
            attrs = self._parse_attrs(m.group(1))
            href = attrs.get('href', '')
            if href and not href.startswith(('#', 'javascript:', 'mailto:')):
                abs_url = self._resolve_url(href, base_url)
                if abs_url:
                    links.append(abs_url)
        return links

    def _extract_headings(self, html):
        headings = []
        for m in re.finditer(r'<(h[1-6])[^>]*>(.*?)</\1>', html, re.IGNORECASE | re.DOTALL):
            level = int(m.group(1)[1])
            text = re.sub(r'<[^>]+>', '', m.group(2)).strip()
            if text:
                headings.append((level, text))
        return headings

    def _resolve_url(self, href, base_url):
        if not base_url:
            return href
        if href.startswith(('http://', 'https://')):
            return href
        return urljoin(base_url, href)


# ---------- Robots.txt Parser ----------

class RobotsParser:
    """Parse and check robots.txt rules."""

    def __init__(self):
        self._rules = {}  # domain -> {user_agent: [rules]}

    def parse(self, domain, robots_txt):
        """Parse robots.txt content for a domain."""
        rules = defaultdict(lambda: {'allow': [], 'disallow': [], 'crawl_delay': None, 'sitemaps': []})
        current_agent = '*'
        for line in robots_txt.split('\n'):
            line = line.split('#')[0].strip()
            if not line:
                continue
            if ':' not in line:
                continue
            key, _, value = line.partition(':')
            key = key.strip().lower()
            value = value.strip()
            if key == 'user-agent':
                current_agent = value.lower()
            elif key == 'disallow' and value:
                rules[current_agent]['disallow'].append(value)
            elif key == 'allow' and value:
                rules[current_agent]['allow'].append(value)
            elif key == 'crawl-delay':
                try:
                    rules[current_agent]['crawl_delay'] = float(value)
                except ValueError:
                    pass
            elif key == 'sitemap':
                rules[current_agent]['sitemaps'].append(value)
        self._rules[domain] = dict(rules)

    def is_allowed(self, url, user_agent='*'):
        """Check if URL is allowed for user agent."""
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path.split('/')[0]
        path = parsed.path or '/'
        if domain not in self._rules:
            return True
        rules = self._rules[domain]
        # Check specific agent first, then wildcard
        agent_key = user_agent.lower()
        agent_rules = rules.get(agent_key, rules.get('*', {'allow': [], 'disallow': []}))

        # Check allow/disallow - longer match wins, allow wins ties
        best_match = None
        best_len = -1
        for pattern in agent_rules.get('allow', []):
            if self._path_matches(path, pattern) and len(pattern) >= best_len:
                best_match = 'allow'
                best_len = len(pattern)
        for pattern in agent_rules.get('disallow', []):
            if self._path_matches(path, pattern) and len(pattern) > best_len:
                best_match = 'disallow'
                best_len = len(pattern)
        return best_match != 'disallow'

    def get_crawl_delay(self, domain, user_agent='*'):
        """Get crawl delay for domain."""
        if domain not in self._rules:
            return None
        rules = self._rules[domain]
        agent_key = user_agent.lower()
        agent_rules = rules.get(agent_key, rules.get('*', {}))
        return agent_rules.get('crawl_delay')

    def get_sitemaps(self, domain, user_agent='*'):
        """Get sitemap URLs for domain."""
        if domain not in self._rules:
            return []
        rules = self._rules[domain]
        sitemaps = []
        for agent_rules in rules.values():
            sitemaps.extend(agent_rules.get('sitemaps', []))
        return list(set(sitemaps))

    def _path_matches(self, path, pattern):
        if pattern.endswith('*'):
            return path.startswith(pattern[:-1])
        return path.startswith(pattern)


# ---------- Sitemap Parser ----------

class SiteMapParser:
    """Parse sitemap XML."""

    def parse(self, xml_content):
        """Parse sitemap XML and return list of URL entries."""
        entries = []
        # Handle sitemap index
        for m in re.finditer(r'<sitemap>(.*?)</sitemap>', xml_content, re.DOTALL):
            loc = re.search(r'<loc>(.*?)</loc>', m.group(1))
            if loc:
                entries.append({'loc': loc.group(1).strip(), 'type': 'sitemap'})

        # Handle URL entries
        for m in re.finditer(r'<url>(.*?)</url>', xml_content, re.DOTALL):
            entry = {'type': 'url'}
            loc = re.search(r'<loc>(.*?)</loc>', m.group(1))
            if loc:
                entry['loc'] = loc.group(1).strip()
            lastmod = re.search(r'<lastmod>(.*?)</lastmod>', m.group(1))
            if lastmod:
                entry['lastmod'] = lastmod.group(1).strip()
            priority = re.search(r'<priority>(.*?)</priority>', m.group(1))
            if priority:
                try:
                    entry['priority'] = float(priority.group(1).strip())
                except ValueError:
                    pass
            changefreq = re.search(r'<changefreq>(.*?)</changefreq>', m.group(1))
            if changefreq:
                entry['changefreq'] = changefreq.group(1).strip()
            if 'loc' in entry:
                entries.append(entry)
        return entries


# ---------- URL Frontier ----------

class URLFrontier:
    """Priority-based URL frontier for crawl scheduling."""

    def __init__(self, max_size=10000):
        self.max_size = max_size
        self._queues = defaultdict(deque)  # domain -> deque of (priority, url)
        self._seen = set()
        self._domain_timestamps = {}  # domain -> last crawl time
        self._politeness_delay = 1.0  # seconds between requests to same domain

    def add(self, url, priority=0.5):
        """Add URL to frontier. Returns True if added."""
        normalized = self._normalize(url)
        if normalized in self._seen:
            return False
        if len(self._seen) >= self.max_size:
            return False
        self._seen.add(normalized)
        domain = self._get_domain(normalized)
        self._queues[domain].append((priority, normalized))
        return True

    def add_many(self, urls, priority=0.5):
        """Add multiple URLs."""
        added = 0
        for url in urls:
            if self.add(url, priority):
                added += 1
        return added

    def get_next(self):
        """Get next URL to crawl, respecting politeness."""
        now = time.time()
        best_url = None
        best_priority = -1
        best_domain = None
        for domain, queue in list(self._queues.items()):
            if not queue:
                continue
            last_crawl = self._domain_timestamps.get(domain, 0)
            if now - last_crawl < self._politeness_delay:
                continue
            priority, url = queue[0]
            if priority > best_priority:
                best_priority = priority
                best_url = url
                best_domain = domain
        if best_url is not None:
            self._queues[best_domain].popleft()
            if not self._queues[best_domain]:
                del self._queues[best_domain]
            self._domain_timestamps[best_domain] = now
            return best_url
        return None

    def get_next_immediate(self):
        """Get next URL ignoring politeness delays (for testing)."""
        best_url = None
        best_priority = -1
        best_domain = None
        for domain, queue in list(self._queues.items()):
            if not queue:
                continue
            priority, url = queue[0]
            if priority > best_priority:
                best_priority = priority
                best_url = url
                best_domain = domain
        if best_url is not None:
            self._queues[best_domain].popleft()
            if not self._queues[best_domain]:
                del self._queues[best_domain]
            return best_url
        return None

    @property
    def size(self):
        return sum(len(q) for q in self._queues.values())

    @property
    def seen_count(self):
        return len(self._seen)

    def has_seen(self, url):
        return self._normalize(url) in self._seen

    def set_politeness_delay(self, delay):
        self._politeness_delay = delay

    def _normalize(self, url):
        url = url.rstrip('/')
        url = url.split('#')[0]  # Remove fragment
        return url

    def _get_domain(self, url):
        parsed = urlparse(url)
        return parsed.netloc or url.split('/')[0]


# ---------- PageRank ----------

class PageRank:
    """PageRank algorithm for link-based ranking."""

    def __init__(self, damping=0.85, iterations=20, tolerance=1e-6):
        self.damping = damping
        self.iterations = iterations
        self.tolerance = tolerance
        self._graph = defaultdict(set)     # url -> set of outgoing urls
        self._in_links = defaultdict(set)  # url -> set of incoming urls
        self._scores = {}

    def add_link(self, from_url, to_url):
        """Add a directed link."""
        self._graph[from_url].add(to_url)
        self._in_links[to_url].add(from_url)
        # Ensure both nodes exist
        if to_url not in self._graph:
            self._graph[to_url]
        if from_url not in self._in_links:
            self._in_links[from_url]

    def add_links(self, from_url, to_urls):
        """Add multiple outgoing links from a page."""
        for to_url in to_urls:
            self.add_link(from_url, to_url)

    def compute(self):
        """Compute PageRank scores. Returns dict of url -> score."""
        nodes = set(self._graph.keys()) | set(self._in_links.keys())
        if not nodes:
            return {}
        n = len(nodes)
        node_list = sorted(nodes)
        scores = {url: 1.0 / n for url in node_list}

        for _ in range(self.iterations):
            new_scores = {}
            dangling_sum = sum(
                scores[url] for url in node_list if not self._graph[url]
            )
            for url in node_list:
                rank = (1 - self.damping) / n
                rank += self.damping * dangling_sum / n
                for in_url in self._in_links[url]:
                    out_degree = len(self._graph[in_url])
                    if out_degree > 0:
                        rank += self.damping * scores[in_url] / out_degree
                new_scores[url] = rank

            # Check convergence
            diff = sum(abs(new_scores[u] - scores[u]) for u in node_list)
            scores = new_scores
            if diff < self.tolerance:
                break

        self._scores = scores
        return scores

    def get_score(self, url):
        """Get PageRank score for a URL."""
        return self._scores.get(url, 0.0)

    def get_top(self, k=10):
        """Get top-k pages by PageRank."""
        return sorted(self._scores.items(), key=lambda x: x[1], reverse=True)[:k]


# ---------- Web Crawler ----------

class WebCrawler:
    """Web crawler with simulated network for testing."""

    def __init__(self, max_pages=1000):
        self.max_pages = max_pages
        self.parser = HTMLParser()
        self.robots = RobotsParser()
        self.sitemap_parser = SiteMapParser()
        self.frontier = URLFrontier(max_size=max_pages)
        self.page_rank = PageRank()
        self._pages = {}       # url -> parsed page data
        self._network = {}     # url -> html content (simulated)
        self._crawl_count = 0
        self.user_agent = 'AgentZeroBot/1.0'

    def add_to_network(self, url, html):
        """Add a page to the simulated network."""
        self._network[url] = html

    def add_robots(self, domain, robots_txt):
        """Add robots.txt for a domain."""
        self.robots.parse(domain, robots_txt)

    def fetch(self, url):
        """Fetch a URL from the simulated network."""
        if url in self._network:
            return self._network[url]
        return None

    def crawl_page(self, url):
        """Crawl a single page. Returns parsed data or None."""
        if not self.robots.is_allowed(url, self.user_agent):
            return None
        html = self.fetch(url)
        if html is None:
            return None
        parsed = self.parser.parse(html, url)
        parsed['url'] = url
        parsed['crawled_at'] = time.time()
        self._pages[url] = parsed
        self._crawl_count += 1

        # Add outgoing links to PageRank
        self.page_rank.add_links(url, parsed['links'])

        return parsed

    def crawl(self, seed_urls, max_pages=None):
        """Crawl starting from seed URLs. Returns list of crawled pages."""
        max_pages = max_pages or self.max_pages
        for url in seed_urls:
            self.frontier.add(url, priority=1.0)

        crawled = []
        while len(crawled) < max_pages:
            url = self.frontier.get_next_immediate()
            if url is None:
                break
            page = self.crawl_page(url)
            if page is None:
                continue
            crawled.append(page)
            # Add discovered links to frontier
            for link in page['links']:
                self.frontier.add(link, priority=0.5)

        # Compute PageRank after crawl
        if crawled:
            self.page_rank.compute()

        return crawled

    @property
    def crawled_pages(self):
        return dict(self._pages)

    @property
    def crawl_count(self):
        return self._crawl_count


# ---------- Search Result ----------

class SearchResult:
    """Rich search result with metadata."""

    def __init__(self, url, title, snippet, score, page_rank=0.0, relevance=0.0):
        self.url = url
        self.title = title
        self.snippet = snippet
        self.score = score
        self.page_rank = page_rank
        self.relevance = relevance

    def to_dict(self):
        return {
            'url': self.url,
            'title': self.title,
            'snippet': self.snippet,
            'score': round(self.score, 4),
            'page_rank': round(self.page_rank, 6),
            'relevance': round(self.relevance, 4),
        }

    def __repr__(self):
        return f"SearchResult(url={self.url!r}, title={self.title!r}, score={self.score:.4f})"


# ---------- AutoComplete ----------

class AutoComplete:
    """Prefix-based query suggestion using a trie."""

    def __init__(self):
        self._trie = {}
        self._counts = defaultdict(int)

    def add_query(self, query, count=1):
        """Record a query for autocomplete."""
        query = query.lower().strip()
        if not query:
            return
        self._counts[query] += count
        node = self._trie
        for ch in query:
            if ch not in node:
                node[ch] = {}
            node = node[ch]
        node['$'] = query

    def suggest(self, prefix, limit=5):
        """Get autocomplete suggestions for prefix."""
        prefix = prefix.lower().strip()
        if not prefix:
            return []
        node = self._trie
        for ch in prefix:
            if ch not in node:
                return []
            node = node[ch]
        # Collect all completions
        completions = []
        self._collect(node, completions)
        # Sort by frequency
        completions.sort(key=lambda q: self._counts[q], reverse=True)
        return completions[:limit]

    def _collect(self, node, results):
        if '$' in node:
            results.append(node['$'])
        for ch, child in node.items():
            if ch != '$':
                self._collect(child, results)

    def get_popular(self, limit=10):
        """Get most popular queries."""
        sorted_queries = sorted(self._counts.items(), key=lambda x: x[1], reverse=True)
        return [(q, c) for q, c in sorted_queries[:limit]]


# ---------- Search Index ----------

class SearchIndex:
    """Unified search index combining IR + PageRank."""

    def __init__(self, text_weight=0.7, pagerank_weight=0.3, use_stemming=True):
        self.text_weight = text_weight
        self.pagerank_weight = pagerank_weight
        self._index = InvertedIndex(use_stemming=use_stemming, remove_stopwords=True)
        self._zone_index = ZoneIndex(
            zone_weights={'title': 3.0, 'headings': 2.0, 'body': 1.0, 'meta': 1.5},
            use_stemming=use_stemming, remove_stopwords=True
        )
        self._url_to_docid = {}
        self._docid_to_url = {}
        self._page_data = {}  # url -> {title, text, headings, meta, links}
        self._pagerank = {}   # url -> score
        self._next_id = 0
        self._snippet_gen = None
        self._spell_corrector = None
        self._autocomplete = AutoComplete()

    def add_page(self, url, page_data):
        """Index a crawled page."""
        doc_id = self._next_id
        self._next_id += 1
        self._url_to_docid[url] = doc_id
        self._docid_to_url[doc_id] = url
        self._page_data[url] = page_data

        # Add to main index
        text = page_data.get('text', '')
        title = page_data.get('title', '')
        full_text = f"{title} {text}"
        doc = Document(doc_id, full_text)
        self._index.add_document(doc)

        # Add to zone index
        headings_text = ' '.join(h[1] for h in page_data.get('headings', []))
        meta_text = ' '.join(page_data.get('meta', {}).values())
        zones = {
            'title': title,
            'headings': headings_text,
            'body': text,
            'meta': meta_text,
        }
        self._zone_index.add_document(doc_id, zones)

        # Reset helpers (they cache vocab)
        self._snippet_gen = None
        self._spell_corrector = None

    def set_pagerank_scores(self, scores):
        """Set PageRank scores from crawler."""
        self._pagerank = dict(scores)

    def search(self, query, top_k=10, use_zones=True):
        """Search and return ranked SearchResult list."""
        if not self._url_to_docid:
            return []

        # Get text relevance scores
        if use_zones:
            text_results = self._zone_index.search(query, top_k=top_k * 3)
        else:
            text_results = self._index.bm25_search(query, top_k=top_k * 3)

        if not text_results:
            return []

        # Normalize text scores
        max_text = max(s for _, s in text_results) if text_results else 1.0
        if max_text == 0:
            max_text = 1.0

        # Normalize PageRank scores
        max_pr = max(self._pagerank.values()) if self._pagerank else 1.0
        if max_pr == 0:
            max_pr = 1.0

        # Combine scores
        results = []
        for doc_id, text_score in text_results:
            url = self._docid_to_url.get(doc_id)
            if url is None:
                continue
            norm_text = text_score / max_text
            pr_score = self._pagerank.get(url, 0.0)
            norm_pr = pr_score / max_pr
            combined = self.text_weight * norm_text + self.pagerank_weight * norm_pr

            page = self._page_data.get(url, {})
            title = page.get('title', url)
            snippet = self._generate_snippet(doc_id, query)

            result = SearchResult(
                url=url,
                title=title,
                snippet=snippet,
                score=combined,
                page_rank=pr_score,
                relevance=text_score,
            )
            results.append(result)

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _generate_snippet(self, doc_id, query):
        """Generate a highlighted snippet for a result."""
        if self._snippet_gen is None:
            self._snippet_gen = SnippetGenerator(self._index, snippet_length=150)
        try:
            return self._snippet_gen.generate(doc_id, query)
        except Exception:
            return ""

    def spell_check(self, query):
        """Suggest corrected query."""
        if self._spell_corrector is None:
            self._spell_corrector = SpellCorrector(self._index)
        return self._spell_corrector.correct_query(query)

    def add_query_history(self, query):
        """Record a search query for autocomplete."""
        self._autocomplete.add_query(query)

    def autocomplete(self, prefix, limit=5):
        """Get autocomplete suggestions."""
        return self._autocomplete.suggest(prefix, limit)

    @property
    def page_count(self):
        return len(self._url_to_docid)

    def get_page(self, url):
        return self._page_data.get(url)

    def has_page(self, url):
        return url in self._url_to_docid


# ---------- Query Processor ----------

class QueryProcessor:
    """Process search queries with parsing and enhancement."""

    def __init__(self, search_index):
        self._index = search_index

    def process(self, raw_query):
        """Process a raw query string. Returns processed query dict."""
        query = raw_query.strip()
        result = {
            'original': query,
            'processed': query,
            'is_phrase': False,
            'terms': [],
            'spell_corrected': None,
        }

        # Detect phrase query (quoted)
        if query.startswith('"') and query.endswith('"') and len(query) > 2:
            result['is_phrase'] = True
            result['processed'] = query[1:-1]

        result['terms'] = tokenize(result['processed'])

        # Spell correction
        corrected = self._index.spell_check(result['processed'])
        if corrected.lower() != result['processed'].lower():
            result['spell_corrected'] = corrected

        return result

    def enhance(self, query_dict, use_correction=False):
        """Enhance query with spell correction if desired."""
        if use_correction and query_dict['spell_corrected']:
            query_dict['processed'] = query_dict['spell_corrected']
            query_dict['terms'] = tokenize(query_dict['processed'])
        return query_dict


# ---------- Crawl Scheduler ----------

class CrawlScheduler:
    """Manage crawl jobs with scheduling."""

    def __init__(self, crawler):
        self.crawler = crawler
        self._jobs = {}  # job_id -> job_info
        self._next_job_id = 1

    def create_job(self, seed_urls, max_pages=100):
        """Create a new crawl job."""
        job_id = self._next_job_id
        self._next_job_id += 1
        self._jobs[job_id] = {
            'id': job_id,
            'seed_urls': seed_urls,
            'max_pages': max_pages,
            'status': 'pending',
            'pages_crawled': 0,
            'created_at': time.time(),
            'started_at': None,
            'completed_at': None,
        }
        return job_id

    def run_job(self, job_id):
        """Execute a crawl job. Returns crawled pages."""
        job = self._jobs.get(job_id)
        if not job:
            return []
        job['status'] = 'running'
        job['started_at'] = time.time()
        pages = self.crawler.crawl(job['seed_urls'], max_pages=job['max_pages'])
        job['pages_crawled'] = len(pages)
        job['status'] = 'completed'
        job['completed_at'] = time.time()
        return pages

    def get_job(self, job_id):
        return self._jobs.get(job_id)

    def list_jobs(self):
        return list(self._jobs.values())


# ---------- Search Engine (Orchestrator) ----------

class SearchEngine:
    """Main search engine orchestrating crawling, indexing, and search."""

    def __init__(self, text_weight=0.7, pagerank_weight=0.3):
        self.crawler = WebCrawler()
        self.index = SearchIndex(text_weight=text_weight, pagerank_weight=pagerank_weight)
        self.scheduler = CrawlScheduler(self.crawler)
        self.query_processor = QueryProcessor(self.index)
        self._search_history = []

    def add_page(self, url, html):
        """Add a page to the network and index it."""
        self.crawler.add_to_network(url, html)
        page = self.crawler.crawl_page(url)
        if page:
            self.index.add_page(url, page)
        return page

    def add_pages(self, pages_dict):
        """Add multiple pages. pages_dict: {url: html}."""
        for url, html in pages_dict.items():
            self.crawler.add_to_network(url, html)
        # Crawl all
        crawled = []
        for url in pages_dict:
            page = self.crawler.crawl_page(url)
            if page:
                self.index.add_page(url, page)
                crawled.append(page)
        # Compute PageRank
        self.crawler.page_rank.compute()
        self.index.set_pagerank_scores(self.crawler.page_rank._scores)
        return crawled

    def crawl_and_index(self, seed_urls, max_pages=100):
        """Crawl from seeds and index all discovered pages."""
        pages = self.crawler.crawl(seed_urls, max_pages=max_pages)
        for page in pages:
            self.index.add_page(page['url'], page)
        # Set PageRank scores
        self.index.set_pagerank_scores(self.crawler.page_rank._scores)
        return pages

    def search(self, query, top_k=10):
        """Search the index. Returns list of SearchResult."""
        # Process query
        query_dict = self.query_processor.process(query)
        # Record for autocomplete
        self.index.add_query_history(query)
        self._search_history.append(query)
        # Search
        results = self.index.search(query_dict['processed'], top_k=top_k)
        return results

    def search_with_info(self, query, top_k=10):
        """Search and return results plus query info."""
        query_dict = self.query_processor.process(query)
        self.index.add_query_history(query)
        self._search_history.append(query)
        results = self.index.search(query_dict['processed'], top_k=top_k)
        return {
            'query': query_dict,
            'results': results,
            'total': len(results),
        }

    def suggest(self, prefix, limit=5):
        """Get autocomplete suggestions."""
        return self.index.autocomplete(prefix, limit)

    @property
    def page_count(self):
        return self.index.page_count


# ---------- Search API (HTTP) ----------

class SearchAPI:
    """REST API for the search engine over HTTP."""

    def __init__(self, engine, host="127.0.0.1", port=0):
        self.engine = engine
        self.router = Router()
        self._setup_routes()
        self._setup_cors()
        self.server = HTTPServer(self.router, host=host, port=port)

    def _setup_cors(self):
        """Add CORS middleware."""
        def cors_middleware(req, next_fn):
            if req.method == 'OPTIONS':
                resp = Response(status=204)
                resp.headers['Access-Control-Allow-Origin'] = '*'
                resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
                resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
                return resp
            resp = next_fn()
            if resp:
                resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp
        self.router.use(cors_middleware)

    def _setup_routes(self):
        """Register all API routes."""
        self.router.get('/api/search', self._handle_search)
        self.router.get('/api/suggest', self._handle_suggest)
        self.router.post('/api/index', self._handle_index)
        self.router.post('/api/crawl', self._handle_crawl)
        self.router.get('/api/stats', self._handle_stats)
        self.router.get('/api/page/:url', self._handle_get_page)
        self.router.get('/api/health', self._handle_health)
        self.router.get('/api/jobs', self._handle_list_jobs)
        self.router.get('/api/jobs/:id', self._handle_get_job)

    def _handle_search(self, req):
        """GET /api/search?q=query&limit=10"""
        q = req.query.get('q', [''])[0] if isinstance(req.query.get('q'), list) else req.query.get('q', '')
        limit = int(req.query.get('limit', ['10'])[0] if isinstance(req.query.get('limit'), list) else req.query.get('limit', '10'))
        if not q:
            return Response.json_response({'error': 'Missing query parameter q'}, status=400)
        info = self.engine.search_with_info(q, top_k=limit)
        return Response.json_response({
            'query': info['query']['original'],
            'processed_query': info['query']['processed'],
            'spell_correction': info['query']['spell_corrected'],
            'total': info['total'],
            'results': [r.to_dict() for r in info['results']],
        })

    def _handle_suggest(self, req):
        """GET /api/suggest?q=prefix&limit=5"""
        q = req.query.get('q', [''])[0] if isinstance(req.query.get('q'), list) else req.query.get('q', '')
        limit = int(req.query.get('limit', ['5'])[0] if isinstance(req.query.get('limit'), list) else req.query.get('limit', '5'))
        suggestions = self.engine.suggest(q, limit=limit)
        return Response.json_response({'suggestions': suggestions})

    def _handle_index(self, req):
        """POST /api/index - Body: {url: string, html: string}"""
        try:
            data = req.json()
        except Exception:
            return Response.json_response({'error': 'Invalid JSON'}, status=400)
        url = data.get('url')
        html = data.get('html')
        if not url or not html:
            return Response.json_response({'error': 'Missing url or html'}, status=400)
        page = self.engine.add_page(url, html)
        if page:
            return Response.json_response({
                'status': 'indexed',
                'url': url,
                'title': page.get('title', ''),
                'links_found': len(page.get('links', [])),
            }, status=201)
        return Response.json_response({'error': 'Failed to index page'}, status=500)

    def _handle_crawl(self, req):
        """POST /api/crawl - Body: {seeds: [url], max_pages: int}"""
        try:
            data = req.json()
            if data is None:
                data = {}
        except Exception:
            return Response.json_response({'error': 'Invalid JSON'}, status=400)
        seeds = data.get('seeds') or []
        max_pages = data.get('max_pages', 100)
        if not seeds:
            return Response.json_response({'error': 'Missing seeds'}, status=400)
        job_id = self.engine.scheduler.create_job(seeds, max_pages=max_pages)
        pages = self.engine.scheduler.run_job(job_id)
        # Index all crawled pages
        for page in pages:
            self.engine.index.add_page(page['url'], page)
        self.engine.crawler.page_rank.compute()
        self.engine.index.set_pagerank_scores(self.engine.crawler.page_rank._scores)
        job = self.engine.scheduler.get_job(job_id)
        return Response.json_response({
            'job_id': job_id,
            'status': job['status'],
            'pages_crawled': job['pages_crawled'],
        }, status=201)

    def _handle_stats(self, req):
        """GET /api/stats"""
        return Response.json_response({
            'pages_indexed': self.engine.page_count,
            'crawl_count': self.engine.crawler.crawl_count,
            'total_jobs': len(self.engine.scheduler.list_jobs()),
        })

    def _handle_get_page(self, req):
        """GET /api/page/:url"""
        url = unquote(req.params.get('url', ''))
        page = self.engine.index.get_page(url)
        if page is None:
            return Response.json_response({'error': 'Page not found'}, status=404)
        return Response.json_response({
            'url': url,
            'title': page.get('title', ''),
            'text': page.get('text', '')[:500],
            'links': page.get('links', []),
        })

    def _handle_health(self, req):
        """GET /api/health"""
        return Response.json_response({'status': 'ok'})

    def _handle_list_jobs(self, req):
        """GET /api/jobs"""
        jobs = self.engine.scheduler.list_jobs()
        return Response.json_response({'jobs': jobs})

    def _handle_get_job(self, req):
        """GET /api/jobs/:id"""
        job_id = int(req.params.get('id', 0))
        job = self.engine.scheduler.get_job(job_id)
        if job is None:
            return Response.json_response({'error': 'Job not found'}, status=404)
        return Response.json_response(job)

    def start(self):
        self.server.start()
        return self.server.port

    def stop(self):
        self.server.stop()
