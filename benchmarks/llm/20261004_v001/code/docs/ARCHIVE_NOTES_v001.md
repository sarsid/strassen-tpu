# Archive implementation notes

The initial local kernel test execution passed all six unittest methods.
Its artifact manifest included interpreter-generated `__pycache__` files that
Git intentionally ignores. The local manifest describes files present locally;
those cache entries are not part of the portable Git artifact. Source, test log
and completion record are retained in Git. No earlier result was altered.

`tools/archive_v002.py` excludes interpreter caches from portable result manifests
and disables bytecode generation for subsequent local executions. This is an
archiving correction, not a change to the kernels or measured test results.
