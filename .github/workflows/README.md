# HCU CI

Changes limited to `docs/**`, Markdown (`*.md`), reStructuredText (`*.rst`),
`LICENSE`, and `NOTICE` skip the HCU build and tests and authorization unit tests.
A PR containing any other file change runs tests normally. The `HCU CI` status
reports documentation-only PRs as successful with an explicit skip description.
The existing Quality Gate compliance and static checks still run.

Pull requests to `main` run HCU CI automatically when the author has `Triage`,
`Write`, `Maintain`, or `Admin` access to this repository. Other contributors need
someone with one of these roles to add the `ready-hcu` label to their PR.

The label authorizes execution on the internal HCU runner. While the label remains,
new commits are tested automatically. Removing the label withdraws authorization
for subsequent runs; it does not stop an already running test. Draft and closed PRs
are not authorized. CI authorization does not approve the PR for merging.

Authorization and status reporting use upstream code on GitHub-hosted runners.
The HCU runner checks out the authorized PR commit and builds and tests it with a
read-only GitHub token and without persisted checkout credentials. The `HCU CI`
commit status links to the workflow logs. For changes requiring tests, it passes
only when the build and tests succeed. Authorization is checked again before
reporting success.

To enable this flow, merge the workflow changes into `main` and create the
`ready-hcu` repository label. To require HCU CI for merging, an administrator can
add `HCU CI` to the required status checks. Existing PRs can trigger the
new workflow by pushing a commit or adding the label.

Pushes to `main` also skip documentation-only changes. Scheduled and manual runs
continue to test their selected repository commit. Test the authorization logic locally with:

```bash
node --test .github/scripts/tests/hcu_authorization.test.cjs
```
