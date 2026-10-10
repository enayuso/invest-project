"""Project-wide domain-error base class.

`server._surface_errors` needs a reliable way to tell "an anticipated
data condition" (bad ticker, stale feed, missing master file) apart from
"a real bug" (a `NaN` reaching `int()`, a broken code path). Using bare
`ValueError`/`RuntimeError` for that boundary does not work: those are
Python's own built-in exception types, raised constantly by pandas/numpy
and the standard library for genuine programming errors, and
`NotImplementedError`/`RecursionError` are literally subclasses of
`RuntimeError`. A tuple keyed on those built-ins would misclassify any
such bug as a data condition.

Every domain error this project defines (`TickerNotFoundError`,
`AmbiguousTickerError`, `StaleDataError`, `NoDataError`,
`KrxMasterError`) subclasses `AdvisorError` instead, and *only*
`AdvisorError`. `server._DOMAIN_ERRORS = (AdvisorError,)` is then the
exact, unambiguous boundary: anything raised by this project on purpose
is an `AdvisorError`; anything else -- including `ValueError`/
`RuntimeError` raised by a third-party library or by a genuine bug in
this project's own code -- falls through to the `내부 오류:` path.
"""

from __future__ import annotations


class AdvisorError(Exception):
    """Base class for every anticipated (non-bug) error this project
    raises on purpose. Never raise this directly -- raise one of its
    subclasses so the client gets a specific, actionable type name."""
