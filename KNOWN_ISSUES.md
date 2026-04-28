# Known Issues

## install.sh / uninstall.sh end-to-end smoke deferred

The Task 4 smoke (install.sh → cross-session prompt → uninstall.sh) was
deliberately not run because the development machine was already in the
post-install state. Scripts were validated by syntax check and byte-match to
spec, but a fresh-machine install has not been exercised end-to-end. First
real test will happen when this is installed on a clean environment; expect
to find and fix issues there.
