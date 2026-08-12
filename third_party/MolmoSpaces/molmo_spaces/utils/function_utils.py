import inspect
from collections.abc import Callable


class _LenientCallable:
    """Picklable callable returned by :func:`make_lenient`."""

    def __init__(self, func: Callable) -> None:
        sig = inspect.signature(func)
        positional: list[str] = []
        kwonly: set[str] = set()
        has_varargs = False
        has_varkw = False
        for p in sig.parameters.values():
            if p.kind is inspect.Parameter.VAR_POSITIONAL:
                has_varargs = True
            elif p.kind is inspect.Parameter.VAR_KEYWORD:
                has_varkw = True
            elif p.kind is inspect.Parameter.KEYWORD_ONLY:
                kwonly.add(p.name)
            else:
                # POSITIONAL_OR_KEYWORD or POSITIONAL_ONLY.
                positional.append(p.name)
        self._func = func
        self._positional_names = positional
        self._kwonly_names = kwonly
        self._all_named = frozenset(positional) | kwonly
        self._has_varargs = has_varargs
        self._has_varkw = has_varkw

    def __call__(self, *args, **kwargs):
        n_pos = len(self._positional_names)
        bound: dict = dict(zip(self._positional_names, args))
        # Leftover positional args are forwarded to func's *args when present,
        # otherwise silently dropped.
        extra_positional = args[n_pos:] if self._has_varargs else ()

        extra_kwargs: dict = {}
        for name, value in kwargs.items():
            if name in self._all_named:
                if name in bound:
                    raise TypeError(
                        f"{getattr(self._func, '__qualname__', self._func)!r} "
                        f"got multiple values for argument {name!r}"
                    )
                bound[name] = value
            elif self._has_varkw:
                # Forward unmatched kwargs to func's **kwargs.
                extra_kwargs[name] = value
            # else: silently dropped

        if extra_positional:
            # When forwarding to *args, named params before *args must be passed
            # positionally. Since extra_positional is non-empty, len(args) > n_pos,
            # so args already covers every positional name. Any kwarg targeting a
            # positional name would have raised in the loop above, so kwonly params
            # (and **kwargs forwards) are the only kwargs left to pass.
            kwonly_for_call = {n: bound[n] for n in self._kwonly_names if n in bound}
            return self._func(*args, **kwonly_for_call, **extra_kwargs)

        return self._func(**bound, **extra_kwargs)


def make_lenient(func: Callable) -> Callable:
    """
    Wrap ``func`` so extra args/kwargs are silently dropped.

    Args matching a declared parameter are forwarded; leftover positional and
    keyword args flow into ``func``'s ``*args`` / ``**kwargs`` when present, and
    are dropped otherwise. Raises ``TypeError`` on double-binding and (via
    Python's normal call machinery) on missing required parameters.

    Args:
        func: The function (or class) to wrap.

    Returns:
        A picklable callable.

    Note:
        Positional-only parameters (after ``/``) are not supported.
    """
    return _LenientCallable(func)
