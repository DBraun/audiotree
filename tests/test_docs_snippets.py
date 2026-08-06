"""Keep the prose documentation honest against the real public API.

The guides under ``docs/source`` are written with ``.. code-block:: python``
rather than doctests, so Sphinx renders them without ever importing audiotree —
a renamed keyword argument leaves the docs build green and every copy-pasted
snippet fatal. These tests close that gap from several directions:

``test_rst_python_blocks_compile``
    Every extracted block has to parse. Catches truncated or mis-indented edits.

``test_rst_python_blocks_use_real_keyword_arguments``
    Static check: every call to a *known* audiotree callable is matched against
    the live :func:`inspect.signature`, so ``ExcerptConfig(enabled=True)`` or
    ``create_balanced_audio_dataset(seed=42)`` fails here even in a snippet that
    could never run for want of a corpus. This is the one that survives renames,
    and it is the only check that reaches all 154 blocks.

``test_rst_cross_references_resolve``
    Every ``:func:``/``:class:``/``:meth:``/``:attr:`` role pointing into
    ``audiotree`` has to name something importable.

``test_docs_build_resolves_audiotree_references`` (``slow``)
    The same roles again, but asked of Sphinx under ``-n``: a guide may not link
    at a dotted path that no ``automodule``/``autoclass`` documents, even when
    Python can import it.

``test_rst_python_blocks_execute`` and ``test_example_scripts_run`` (both
``slow``)
    Actually run what can be run, against a synthetic corpus built from the
    placeholder paths the guides themselves name.

Opt-out markers, for snippets that legitimately cannot be checked:

``.. skip-snippet-test: reason``
    An rst comment directly above a block (blank lines between are fine). Skips
    that block in *every* test here. Use sparingly and always with a reason.

``.. skip-snippet-exec: reason``
    Same placement, but only skips execution — the block is still parsed and
    still has its keyword arguments checked. This is the right marker for a
    fragment that references undefined names or a real corpus.

``# Before`` / ``# After``
    Inside a block, everything from a ``# Before`` comment up to the next
    ``# After`` comment is exempt from the keyword-argument check. The migration
    guide shows pre-1.0 code side by side with its replacement; the pre-1.0 half
    is *supposed* to name parameters that no longer exist, and the 1.0 half
    stays checked.
"""

import ast
import inspect
import pathlib
import re
import subprocess
import sys
import textwrap
import typing

import numpy as np
import pytest
import soundfile

import audiotree
import audiotree.sources
import audiotree.transforms
import audiotree.transforms.jax

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS_SOURCE = REPO_ROOT / "docs" / "source"
EXAMPLES = REPO_ROOT / "examples"

_DIRECTIVE = re.compile(
    r"^(?P<indent>[ \t]*)\.\.[ \t]+"
    r"(?P<name>code-block|testcode|testsetup)"
    r"(?:::[ \t]*(?P<lang>\S+)?)?[ \t]*$"
)
_SKIP_ALL = re.compile(r"^[ \t]*\.\.[ \t]+skip-snippet-test:[ \t]*(?P<reason>.*)$")
_SKIP_EXEC = re.compile(r"^[ \t]*\.\.[ \t]+skip-snippet-exec:[ \t]*(?P<reason>.*)$")
_OPTION = re.compile(r"^[ \t]*:[a-zA-Z-]+:")

# Paths the guides use as stand-ins for a user's corpus. The execution test
# points them all at a real synthetic corpus so the snippets can run unmodified.
_PLACEHOLDER_ROOTS = ("/data", "/fast_storage", "/path/to", "/mnt/data")


class Snippet(typing.NamedTuple):
    """One executable block lifted out of an rst file."""

    path: pathlib.Path
    line: int  # 1-indexed line of the directive
    kind: str  # "code-block", "testcode" or "testsetup"
    source: str  # dedented block body
    skip: typing.Optional[str]  # reason, if fully opted out
    skip_exec: typing.Optional[str]  # reason, if opted out of execution only

    @property
    def id(self) -> str:
        return f"{self.path.relative_to(DOCS_SOURCE)}:{self.line}"


def _preceding_marker(lines: typing.List[str], index: int, pattern: re.Pattern):
    """Return the reason from ``pattern`` directly above ``lines[index]``."""
    i = index - 1
    while i >= 0 and not lines[i].strip():
        i -= 1
    if i < 0:
        return None
    match = pattern.match(lines[i])
    return (match.group("reason").strip() or "no reason given") if match else None


def _extract(path: pathlib.Path) -> typing.List[Snippet]:
    lines = path.read_text().splitlines()
    snippets = []
    for i, line in enumerate(lines):
        directive = _DIRECTIVE.match(line)
        if directive is None:
            continue
        is_python = (
            directive.group("name") != "code-block"
            or directive.group("lang") == "python"
        )
        if not is_python:
            continue
        indent = len(directive.group("indent").expandtabs(8))

        # Body: the contiguous indented run after the directive, minus its
        # options (``:hide:``, ``:emphasize-lines:``, …).
        body = []
        j = i + 1
        while j < len(lines):
            current = lines[j]
            if not current.strip():
                body.append("")
                j += 1
                continue
            if len(current) - len(current.lstrip()) <= indent:
                break
            body.append(current)
            j += 1
        while body and (not body[0].strip() or _OPTION.match(body[0])):
            body.pop(0)
        if not body:
            continue

        snippets.append(
            Snippet(
                path=path,
                line=i + 1,
                kind=directive.group("name"),
                source=textwrap.dedent("\n".join(body)).strip("\n") + "\n",
                skip=_preceding_marker(lines, i, _SKIP_ALL),
                skip_exec=_preceding_marker(lines, i, _SKIP_EXEC),
            )
        )
    return snippets


def _all_snippets() -> typing.List[Snippet]:
    snippets = []
    for rst in sorted(DOCS_SOURCE.rglob("*.rst")):
        snippets.extend(_extract(rst))
    assert snippets, f"no python blocks found under {DOCS_SOURCE}"
    return snippets


SNIPPETS = _all_snippets()
CHECKED = [s for s in SNIPPETS if s.skip is None]


def _registry() -> typing.Dict[str, typing.Callable]:
    """Public callables a snippet may name, keyed by the name it would use."""
    modules = (
        audiotree,
        audiotree.sources,
        audiotree.transforms,
        audiotree.transforms.jax,
    )
    registry = {}
    for module in modules:
        for name in getattr(module, "__all__", ()):
            obj = getattr(module, name)
            if callable(obj):
                registry.setdefault(name, obj)
    # Constructors and classmethods reached through the class name, e.g.
    # ``AudioTree.create(...)`` or ``AudioWriter(...)``.
    for cls in (audiotree.AudioTree, audiotree.AudioWriter, audiotree.TreeWriter):
        for name, member in vars(cls).items():
            if name.startswith("_"):
                continue
            member = getattr(cls, name)
            if callable(member):
                registry.setdefault(f"{cls.__name__}.{name}", member)
    return registry


REGISTRY = _registry()


def _exempt_lines(source: str) -> typing.Set[int]:
    """Line numbers inside a ``# Before`` … ``# After`` region."""
    exempt, inside = set(), False
    for lineno, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if re.match(r"#\s*Before\b", stripped):
            inside = True
        elif re.match(r"#\s*After\b", stripped):
            inside = False
        if inside:
            exempt.add(lineno)
    return exempt


def _called_name(node: ast.Call) -> typing.Optional[str]:
    """The registry key for ``node``'s callee, or None if it is not ours.

    Only bare names and ``Class.method`` are resolved. An attribute call on an
    arbitrary expression (``ds.batch(...)``, ``tree.replace_lufs(...)``) is left
    alone: the receiver's type is unknown here, and grain's datasets share
    several method names with :class:`~audiotree.core.AudioTree`.
    """
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return f"{func.value.id}.{func.attr}"
    return None


def _bad_keywords(snippet: Snippet) -> typing.List[str]:
    tree = ast.parse(snippet.source)
    exempt = _exempt_lines(snippet.source)
    # A snippet that rebinds a public name (``volume_norm = argbind.bind(...)``)
    # no longer calls the function we have a signature for.
    rebound = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            rebound.add(node.name)
        elif isinstance(node, ast.Assign):
            rebound.update(t.id for t in node.targets if isinstance(t, ast.Name))

    problems = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or node.lineno in exempt:
            continue
        name = _called_name(node)
        if name is None or name in rebound or name not in REGISTRY:
            continue
        try:
            signature = inspect.signature(REGISTRY[name])
        except (TypeError, ValueError):  # pragma: no cover - builtins etc.
            continue
        parameters = signature.parameters
        if any(p.kind is p.VAR_KEYWORD for p in parameters.values()):
            continue
        for keyword in node.keywords:
            if keyword.arg is None or keyword.arg in parameters:
                continue
            problems.append(
                f"line {node.lineno}: {name}({keyword.arg}=...) is not a "
                f"parameter of {name}{signature}"
            )
    return problems


@pytest.mark.parametrize("snippet", CHECKED, ids=lambda s: s.id)
def test_rst_python_blocks_compile(snippet: Snippet):
    """Every documented python block is at least valid python."""
    compile(snippet.source, snippet.id, "exec")


@pytest.mark.parametrize("snippet", CHECKED, ids=lambda s: s.id)
def test_rst_python_blocks_use_real_keyword_arguments(snippet: Snippet):
    """Documented keyword arguments still exist on the functions they name."""
    problems = _bad_keywords(snippet)
    assert not problems, "\n".join([f"{snippet.id}:"] + problems)


_ROLE = re.compile(
    r":(?:func|class|meth|attr|data|mod):`~?(?P<target>audiotree[\w.]*)`"
)


def _resolves(dotted: str) -> bool:
    parts = dotted.split(".")
    for split in range(len(parts), 0, -1):
        module_name = ".".join(parts[:split])
        try:
            module = __import__(module_name, fromlist=["_"])
        except ImportError:
            continue
        obj = module
        for attribute in parts[split:]:
            try:
                obj = getattr(obj, attribute)
            except AttributeError:
                return False
        return True
    return False


@pytest.mark.parametrize(
    "rst", sorted(DOCS_SOURCE.rglob("*.rst")), ids=lambda p: p.name
)
def test_rst_cross_references_resolve(rst: pathlib.Path):
    """Every ``audiotree`` cross-reference in the prose names a real object."""
    dead = sorted(
        {
            match.group("target")
            for match in _ROLE.finditer(rst.read_text())
            if not _resolves(match.group("target"))
        }
    )
    assert not dead, f"{rst.relative_to(DOCS_SOURCE)} links at missing objects: {dead}"


_NITPICK = re.compile(
    r"^(?P<file>[^:]+\.rst):\d+: WARNING: py:\w+ reference target not found: "
    r"(?P<target>audiotree[\w.]*)"
)


@pytest.mark.slow
def test_docs_build_resolves_audiotree_references(tmp_path):
    """A guide may not link at an ``audiotree`` object no page documents.

    :func:`test_rst_cross_references_resolve` only asks Python whether the
    object exists. Sphinx additionally needs some ``automodule``/``autoclass``
    to have *documented* it under that exact dotted path, and it reports the
    difference only under ``-n``. Restricted to warnings raised from an ``.rst``
    file so that an undocumented reference in a source docstring is somebody
    else's build to fix.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-b",
            "dummy",
            "-n",
            "-q",
            str(DOCS_SOURCE),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT / "docs",
    )
    assert result.returncode == 0, f"docs build failed:\n{result.stderr}"
    dead = sorted(
        {
            f"{pathlib.Path(match.group('file')).name} -> {match.group('target')}"
            for line in result.stderr.splitlines()
            for match in [_NITPICK.match(line.strip())]
            if match is not None
        }
    )
    assert not dead, "cross-references no page documents:\n" + "\n".join(dead)


_PLACEHOLDER_PATH = re.compile(
    r"""["'](?P<path>(?:%s)/[^"'\s]*)["']"""
    % "|".join(re.escape(root) for root in _PLACEHOLDER_ROOTS)
)
_AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg")


def _placeholder_paths() -> typing.Set[str]:
    """Every ``/data/...``-style path the guides mention, in any block."""
    return {
        match.group("path")
        for snippet in SNIPPETS
        for match in _PLACEHOLDER_PATH.finditer(snippet.source)
    }


def _write_wav(path: pathlib.Path, rng: np.random.Generator, seconds: float = 4.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = int(44_100 * seconds)
    soundfile.write(
        path, (0.1 * rng.standard_normal((samples, 2))).astype(np.float32), 44_100
    )


def _build_corpus(root: pathlib.Path) -> pathlib.Path:
    """Materialize a synthetic corpus at every path the guides name.

    Built from the snippets themselves rather than a hand-kept list, so a guide
    that starts talking about ``/data/percussion`` gets a ``/data/percussion``
    without anyone having to remember to add one here.
    """
    rng = np.random.default_rng(0)
    for placeholder in sorted(_placeholder_paths()):
        target = root / placeholder.lstrip("/")
        if target.suffix in _AUDIO_SUFFIXES:
            _write_wav(target, rng)
        else:
            for index in range(3):
                _write_wav(target / f"{index}.wav", rng)
    return root


def _retarget(source: str, corpus: pathlib.Path) -> str:
    """Point the guides' placeholder corpus paths at the synthetic corpus."""
    for root in _PLACEHOLDER_ROOTS:
        source = source.replace(f'"{root}/', f'"{corpus}{root}/')
        source = source.replace(f"'{root}/", f"'{corpus}{root}/")
    return source


def _run_page(page: pathlib.Path, corpus: pathlib.Path) -> None:
    """Execute one page's blocks in a single namespace, the way a reader would.

    Blocks share a namespace within a page so that ``.. testsetup::`` and
    earlier blocks can define the names later ones use — that is how the pages
    are written to be read.
    """
    namespace = {"__name__": "__docs__"}
    for snippet in _extract(page):
        if snippet.skip is not None or snippet.skip_exec is not None:
            continue
        source = _retarget(snippet.source, corpus)
        try:
            exec(compile(source, snippet.id, "exec"), namespace)
        except Exception as error:  # noqa: BLE001 - re-raised with the snippet
            raise AssertionError(
                f"{snippet.id} raised {type(error).__name__}: {error}\n\n{source}"
            ) from error


_PAGES = sorted({s.path for s in SNIPPETS})

#: A page whose snippets all hang or block would otherwise wedge the suite.
#: Grain's repeated datasets are infinite by construction, so a snippet that
#: iterates one has to be opted out rather than merely waited on.
_PAGE_TIMEOUT_SECONDS = 600


@pytest.fixture(scope="session")
def corpus(tmp_path_factory) -> pathlib.Path:
    return _build_corpus(tmp_path_factory.mktemp("docs_corpus"))


@pytest.mark.slow
@pytest.mark.parametrize("page", _PAGES, ids=lambda p: p.name)
def test_rst_python_blocks_execute(page: pathlib.Path, corpus):
    """Every block that does not opt out actually runs.

    In a subprocess, so that a snippet which leaks a grain worker pool or wedges
    on an infinite dataset fails this one page instead of the whole suite.
    """
    if not [s for s in _extract(page) if s.skip is None and s.skip_exec is None]:
        pytest.skip("every block on this page opts out of execution")
    result = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).resolve()), str(page), str(corpus)],
        capture_output=True,
        text=True,
        timeout=_PAGE_TIMEOUT_SECONDS,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr or result.stdout


@pytest.mark.slow
@pytest.mark.parametrize("script", sorted(EXAMPLES.rglob("*.py")), ids=lambda p: p.name)
def test_example_scripts_run(script: pathlib.Path):
    """The example scripts run to completion, not just to import.

    ``--doctest-modules`` imports these, which stops at the ``if __name__ ==
    "__main__"`` guard and so never touched the body that actually calls the
    API.
    """
    if script.name.startswith("test_"):
        pytest.skip("pytest collects this one itself")
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=script.parent,
    )
    assert result.returncode == 0, f"{script.name} failed:\n{result.stderr}"


if __name__ == "__main__":
    # Subprocess entry point for ``test_rst_python_blocks_execute``:
    # ``python tests/test_docs_snippets.py <page.rst> <corpus_dir>``.
    from absl import flags

    # Same reason as tests/conftest.py: grain reads absl flags from inside its
    # multiprocessing prefetch path, and this is not an ``absl.app`` entry point.
    flags.FLAGS.mark_as_parsed()
    _run_page(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
