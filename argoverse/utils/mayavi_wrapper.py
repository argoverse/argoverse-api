# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

"""This module abstracts the usage of mayavi out of most of the code.

Fundamentally, `mayavi` isn't fun to install and some users might not care about that functionality so it is an
optional dependency of this package.  In order to support the case of "maybe we have it, maybe we don't",
we provide this module that wraps mayavi.mlab.

This module aims to be importable, but you'll get a `NoMayaviAvailableError` if you try to use
anything on `mayavi_wrapper.mlab` unless you have installed `mayavi` properly.
"""

from typing import TYPE_CHECKING


class NoMayaviAvailableError(ImportError):
    """Raised when mayavi is not available but we try to use its functionality."""


if TYPE_CHECKING:
    from mayavi import mlab
else:
    try:
        from mayavi import mlab
    except ImportError as err:

        local_err = err  # Need to keep a reference -- otherwise `err` is an undefined name.

        class _MissingMlab:
            """Defer raising ImportError until mayavi.mlab is used rather than when we import it.

            This lets is maintain that mayavi is an optional dependency without working too hard to make sure
            that we didn't accidentally try to import it on a code path that actually doesn't need it.
            """

            def __getattribute__(self, name: str) -> None:
                """Unconditionally raise an exception if we try to use any functionality from mayavi.

                Args:
                    name: The name of the property.
                """
                raise NoMayaviAvailableError from local_err

        mlab = _MissingMlab()
