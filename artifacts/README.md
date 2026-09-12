# artifacts

Exported controllers land here (`neurofly export runs/<run> artifacts/<name>`). They are
git-ignored by default because a real-brain artifact is about 150 MB; `git add -f` the ones
you want to ship. `neurofly-core info <dir>` describes one.
