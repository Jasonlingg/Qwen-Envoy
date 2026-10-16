"""Opt-in argument adapters appended after the unchanged, pinned v1 preamble."""

import hashlib

HARNESS_V2_VERSION = "v2"
V1_TOOL_PREAMBLE_SHA256 = "233adfef3cd72c8b8221e45dcaa58c76809f36bd717d1d342514a7c332c3dd71"

HARNESS_V2_PREAMBLE = '''
# Capture the freshly defined v1 functions on each Docker cumulative replay.
_v1_search, _v1_read, _v1_passage, _v1_search_within = search, read, passage, search_within
_MISSING = object()
_TOP_K_ALIASES = ("top_k", "k", "n", "limit", "max_results", "num_results")
_USAGE = {
    "search": "search(query, top_k=5), or search(doc_id, query, top_k=8) in either order",
    "read": "read(doc_id), or read(doc_id, start, end), or read(doc_id, offset=0, length=1600)",
    "passage": "passage(doc_id, start=0, length=1600), or passage(doc_id, start, end=END)",
    "search_within": "search_within(doc_id, query, top_k=8)",
}

def _usage_error(tool, problem):
    return TypeError(f"{tool}(): {problem}. Usage: {_USAGE[tool]}")

def _as_int(tool, name, value):
    if type(value) is int:
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-+").isdigit():
        try:
            return int(value.strip())
        except ValueError:
            pass
    raise _usage_error(tool, f"{name} must be an integer, got {value!r}")

def _take(tool, args, kwargs, names, default=_MISSING):
    found = [name for name in names if name in kwargs]
    if len(found) > 1 or (found and args):
        raise _usage_error(tool, "give " + names[0] + " only once")
    if found:
        return kwargs.pop(found[0])
    return args.pop(0) if args else default

def _reject_extra(tool, args, kwargs):
    if args or kwargs:
        raise _usage_error(tool, "unexpected arguments " + repr(args) + " " + repr(list(kwargs)))

def _top_k(tool, args, kwargs, default):
    value = _take(tool, args, kwargs, _TOP_K_ALIASES, default)
    return default if value is None else _as_int(tool, "top_k", value)

def _text(tool, name, value):
    if not isinstance(value, str) or not value.strip():
        raise _usage_error(tool, name + " must be non-empty text")
    return value

def search(*args, **kwargs):
    args = list(args)
    doc_id = kwargs.pop("doc_id", _MISSING)
    ids = {doc["doc_id"] for doc in _load_all_docs()}
    known = lambda value: isinstance(value, str) and value in ids
    if doc_id is _MISSING and args:
        if any(key in kwargs for key in ("query", "q")) and known(args[0]):
            doc_id = args.pop(0)
        elif len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], str):
            first, second = known(args[0]), known(args[1])
            if first and second:
                raise _usage_error("search", "two document IDs are ambiguous; use doc_id= and query=")
            if first or second:
                doc_id = args.pop(0 if first else 1)
    query = _text("search", "query", _take("search", args, kwargs, ("query", "q")))
    top_k = _top_k("search", args, kwargs, 5 if doc_id is _MISSING else 8)
    method = _take("search", args, kwargs, ("method",), "keyword")
    _reject_extra("search", args, kwargs)
    if doc_id is not _MISSING:
        if method != "keyword":
            raise _usage_error("search", "method is only supported for corpus discovery")
        return search_within(doc_id, query, top_k=top_k)
    return _v1_search(query, top_k=top_k, method=method)

def read(*args, **kwargs):
    args = list(args)
    doc_id = _text("read", "doc_id", _take("read", args, kwargs, ("doc_id",)))
    start = _take("read", args, kwargs, ("start", "offset"), None)
    end = _take("read", args, kwargs, ("end",), None)
    length = kwargs.pop("length", None)
    _reject_extra("read", args, kwargs)
    if start is None and end is None and length is None:
        return _v1_read(doc_id)
    return passage(doc_id, 0 if start is None else start, length=length, end=end)

def passage(*args, **kwargs):
    args = list(args)
    doc_id = _text("passage", "doc_id", _take("passage", args, kwargs, ("doc_id",)))
    start = _take("passage", args, kwargs, ("start", "offset"), 0)
    length = _take("passage", args, kwargs, ("length", "size"), None)
    end = kwargs.pop("end", None)
    _reject_extra("passage", args, kwargs)
    start = _as_int("passage", "start", start)
    if end is not None:
        if length is not None:
            raise _usage_error("passage", "give length or end, not both")
        length = _as_int("passage", "end", end) - start
    length = 1600 if length is None else _as_int("passage", "length", length)
    if length < 1:
        raise _usage_error("passage", "length must be positive; end must follow start")
    return _v1_passage(doc_id, start, min(length, 3000))

def search_within(*args, **kwargs):
    args = list(args)
    doc_id = _text("search_within", "doc_id", _take("search_within", args, kwargs, ("doc_id",)))
    query = _text("search_within", "query", _take("search_within", args, kwargs, ("query", "q")))
    top_k = _top_k("search_within", args, kwargs, 8)
    _reject_extra("search_within", args, kwargs)
    return _v1_search_within(doc_id, query, top_k=top_k)

def submit(*args, **kwargs):
    raise RuntimeError(
        "submit() must be the only statement in its action, with literal values, e.g. "
        "submit(answer='...', citations=['doc_id'], "
        "evidence=[{'doc_id': 'doc_id', 'start': 0, 'end': 100}])"
    )
'''

HARNESS_V2_SHA256 = hashlib.sha256(HARNESS_V2_PREAMBLE.encode()).hexdigest()

V2_PROMPT_ADDENDUM = (
    '\n\nHarness v2: finish with a SUBMIT line or submit(answer="...", '
    'citations=["doc_id"], evidence=[{"doc_id": "doc_id", "start": 0, "end": 100}]) '
    'as the only statement in a turn, using literal values. To abstain, use '
    'submit(answer="Unanswerable", citations=[], evidence=[]). Corpus search(query) '
    'defaults to 5 documents; search_within(doc_id, query) and paper-scoped '
    'search(doc_id, query) or search(query, doc_id) default to 8 windows. Discover '
    'document IDs with corpus search() before using paper-scoped tools. '
    'Tools accept common argument aliases and numeric strings and return usage '
    'messages when misused. A valid quote establishes provenance, not semantic support.'
)
