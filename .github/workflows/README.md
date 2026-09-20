# HCU CI

Pull requests to `main` use `pull_request` and test the temporary merge of the PR
with its base branch. Push, scheduled, and manual runs test their selected commit.
The HCU build and tests share one device and run serially. The concurrency group
keeps one running and one pending run; a new run replaces the pending run.

PR authors with `Triage`, `Write`, `Maintain`, or `Admin` access to this repository
are automatically authorized. Other authors need a user with one of these roles
to apply `ready-hcu`. Organization membership alone does not grant hardware access.
The label remains valid for new commits until removed. For authors requiring the
label, removing it prevents later runs and a passing final check; it does not
terminate an already running test.
Draft, closed, and outdated PR events do not authorize hardware execution.

Changes limited to `docs/**`, Markdown (`*.md`), reStructuredText (`*.rst`),
`LICENSE`, and `NOTICE` skip HCU build and tests and authorization unit tests.
The `HCU CI` job reports the documentation-only skip as success. Quality Gate
checks still run. For other changes, `HCU CI` passes only after authorization,
build, and tests succeed. All workflow jobs use read permissions; the aggregate
result is an Actions check. Authorization is checked again before reporting success.

GitHub's repository approval policy may require **Approve workflows** before a
fork PR workflow starts. `ready-hcu` does not bypass that policy and does not
approve the PR for merging. The workflow and helpers are part of the PR merge;
label checks are an execution policy, not an isolation boundary against modified
workflows. Maintainers must review CI changes before allowing them to run on the
privileged, persistent HCU runner.

The base image comes from repository variable `IMAGE_URL`.
`HCU_PIP_INDEX_URL` and `HCU_PIP_TRUSTED_HOST` are optional.

An administrator can require the `HCU CI` Actions check in the branch rules.
After changing the workflow, PR branches must include the updated workflow.
GitHub event delivery, fork token permissions, runner configuration, and an actual
HCU build and test run must be verified on GitHub before relying on the check.

Run authorization tests from the repository root with Node.js:

```bash
node --test .github/scripts/tests/hcu_authorization.test.cjs
```
