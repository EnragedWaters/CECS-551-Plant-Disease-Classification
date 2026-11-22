# Contributing without breaking the shared project

Commits save your work. GitHub checks decide whether that work can merge into
`main`. You can commit and push unfinished work to a branch; no local commit
hook is required or installed.

## The normal workflow

1. Start from the latest `main` and make a branch for your change.
2. Commit and push to that branch. Prefer a branch in this shared repository.
3. Open a **draft pull request early**, with a short explanation of what is being
   changed and what remains. This makes work visible while it is unfinished.
4. When ready, mark the pull request ready for review. Fix any failed checks and
   push again; checks rerun automatically.
5. Update the branch if `main` has moved, resolve conflicts, and wait for the new
   checks. Merge only when they pass, then remove the completed branch.

```bash
git switch main
git pull --ff-only
git switch -c describe-your-change
# Edit and commit your files.
git push -u origin describe-your-change
```

These commands assume your existing work is committed or safely set aside first.
Do not reset or discard someone else's changes to get a clean working tree.

Shared branches and early draft pull requests are **team conventions**, not rules
GitHub can enforce. Forks remain valid; pull requests from forks must pass the
same required checks. Unpushed work and private fork activity are not made visible
by these checks. A fork contributor's first workflow run may need approval on
GitHub before the checks can start; do not bypass them while waiting.

## What GitHub enforces on main

- Changes must arrive through a pull request, not a direct push.
- `Project checks (ubuntu-latest)` and `Project checks (windows-latest)` must pass.
- The branch must be current with `main` before merging. A new commit needs new checks.
- Force pushes and branch deletion are blocked.
- No routine bypass is configured, including for the owner.

No separate reviewer approval is required for every pull request. This lets the
owner merge their own passing work without waiting for another team member.
The owner retains repository administration access to repair a genuinely broken
ruleset. That is a recovery option, not the normal way around a failing check.
A failing pull request can always be corrected, pushed again, or closed.

## Why is GitHub refusing my change?

| What happened | What to do |
| --- | --- |
| A local commit failed | Read Git's error: check your selected files, commit message, identity or editor. These remote checks do not block local commits. |
| A push to `main` was rejected | Keep the commits. Create a branch at your current commit, push that branch and open a pull request. Do not force-push. |
| A pull request has a red check | Open **Checks → failed job → failed step**. Fix the reported file/test, commit to the same branch and push again. |
| Checks are pending or awaiting approval | Wait for the run, or ask the owner to approve an eligible fork run. A missing check is not a pass. |
| Checks passed but the branch is behind `main` | Merge the latest `origin/main` into your branch, resolve conflicts and push. Wait for checks again. |
| A branch has a red check | Your work is still saved. Only merging to `main` is blocked. |

## Run the same checks locally

Use Python 3.12 and Node.js 22 (LTS). In a dedicated test environment, install:

```bash
python -m pip install -r requirements-checks-torch.txt
python -m pip install -r requirements-checks.txt
python -m playwright install --with-deps chromium --only-shell
```

The first file installs the **CPU** PyTorch build. Use a separate virtual
environment to keep an existing GPU research environment unchanged. An existing
compatible environment can run checks too; the checker selects CPU execution.
On Linux, installing Chromium's system libraries may require administrator
access. Windows installs the browser without WSL.

Then run the one command used on GitHub:

```bash
python scripts/check_project.py
```

It checks Python syntax, runtime package imports, JavaScript syntax, JSON and
notebook structure, category/demo evidence consistency, submission copies and
tracked-file hygiene. Then it runs the unit tests and a headless browser exercise
of the real demo and recorded replay with invented images and tiny fixed models.
Windows runs the real PowerShell/CMD wrappers against a stand-in Python entry;
Linux runs the shell wrappers. Tests cover paths with spaces, argument forwarding,
exit handling and failure messages. Only the applicable OS wrapper tests run locally;
GitHub runs both operating systems.

The browser exercises both dataset panels, all seven stages, lessons,
play/pause/stepping, request failure recovery, sample counts, narrow layouts and
offline replay with HTTP blocked. It never opens a visible browser. Temporary
fixture outputs are removed. Unit tests may perform tiny synthetic optimization
steps; they never retrain the research models or load the real datasets.

`--static-only` is a quick optional check while editing. It is not a substitute
for the full command or the required GitHub jobs. Notebook cells are not executed
against research datasets in automation. Optional presentation-authoring scripts
receive syntax checks; real slide rendering and saved-model verification remain
separate checks documented in the presentation guide.

## Files and evidence

Keep raw datasets, model weights, local settings and generated experiment outputs
outside Git. Existing credited slide illustrations and report figures are allowed.
The checker catches common paths, filenames, weight/archive extensions and files
over 10 MiB. It is not a comprehensive secret scanner and cannot identify every
private document from its contents. Inspect the files you stage.

Refresh submission copies when intentionally updating their source presentation.
Preserve the original paper and result records. Put proposed paper revisions and
new experiments in separate locations; do not overwrite original evidence just
to satisfy a consistency check.

## Changing the checks themselves

Explain why a workflow, test, guard, dependency pin or protection change is needed
in its pull request, and show that it still catches the problem it guards against.
Ask the owner to review those changes; do not quietly remove tests, rename required
jobs or add skips to turn a failing pull request green. Required job names must
match the GitHub ruleset. This review request is a team convention, not an enforced
approval gate. The protections prevent accidental breakage, not deliberate changes
to the checks by a trusted collaborator or administrator.

Use read-only workflow permissions. Do not run fork code with repository secrets,
use `pull_request_target` to execute contributor code, or install automation on the
owner's personal machine as a self-hosted runner. Current jobs use GitHub-hosted
Windows/Linux runners and CPU fixtures. Ordinary pushes to feature branches do not
start duplicate jobs; opening/updating their pull request runs the checks, including
draft pull requests. A merge also checks `main`.

Passing checks is evidence, not a guarantee: they cannot prove scientific claims,
real-data accuracy, visual presentation quality, every GPU/OS setup, or timely team
communication. Report those checks separately when relevant.
