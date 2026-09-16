// Copyright 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

const assert = require("node:assert/strict");
const { test } = require("node:test");
const { authorize, prepare, finish } = require("../hcu_authorization.cjs");

function fixture({ permission = "read", labels = [], events = [], draft = false,
  state = "open", currentSha = "pr-sha", permissions = {}, apiError = false,
  files = [{ filename: "turbo_physai/__init__.py" }], changedFiles = files.length } = {}) {
  const outputs = {};
  const statuses = [];
  const failures = [];
  const permissionRequests = [];
  const context = {
    repo: { owner: "example", repo: "project" },
    sha: "base-sha",
    runId: 42,
    serverUrl: "https://github.com",
    payload: { pull_request: { number: 9, head: { sha: "pr-sha" } } },
  };
  const github = {
    rest: {
      pulls: { get: async () => ({ data: {
        number: 9, state, draft, head: { sha: currentSha },
        changed_files: changedFiles,
        user: { login: "contributor" }, labels: labels.map((name) => ({ name })),
      } }), listFiles: Symbol("listFiles") },
      repos: {
        getCollaboratorPermissionLevel: async ({ username }) => {
          permissionRequests.push(username);
          if (apiError) throw new Error("GitHub API unavailable");
          return { data: { permission: permissions[username] ?? permission } };
        },
        createCommitStatus: async (value) => { statuses.push(value); },
      },
      issues: { listEvents: Symbol("listEvents") },
    },
    paginate: async (method, options) => {
      if (method === github.rest.pulls.listFiles) {
        assert.equal(options.pull_number, 9);
        return files;
      }
      assert.equal(method, github.rest.issues.listEvents);
      assert.equal(options.issue_number, 9);
      return events;
    },
  };
  const core = {
    setOutput: (key, value) => { outputs[key] = value; },
    info: () => {},
    setFailed: (message) => { failures.push(message); },
  };
  return { github, context, core, outputs, statuses, failures, permissionRequests };
}

const labeled = (actor, event = "labeled") => ({
  event, actor: { login: actor }, label: { name: "ready-hcu" },
});

for (const permission of ["admin", "maintain", "write", "triage"]) {
  test(`${permission} contributor is automatically authorized`, async () => {
    const args = fixture({ permission });
    await prepare(args);
    assert.equal(args.outputs.authorized, "true");
    assert.equal(args.outputs.sha, "pr-sha");
    assert.equal(args.statuses[0].state, "pending");
    assert.equal(args.statuses[0].sha, "pr-sha");
    assert.equal(args.statuses[0].context, "HCU CI");
    assert.equal(args.statuses[0].target_url, "https://github.com/example/project/actions/runs/42");
  });
}

for (const permission of ["read", "none", "unknown"]) {
  test(`${permission} contributor requires authorization`, async () => {
    const args = fixture({ permission });
    await prepare(args);
    assert.equal(args.outputs.authorized, "false");
    assert.equal(args.statuses[0].state, "failure");
  });
}

test("external contributor is authorized by a trusted label actor", async () => {
  const args = fixture({ labels: ["ready-hcu"], events: [labeled("maintainer")],
    permissions: { maintainer: "maintain" } });
  assert.equal((await authorize(args)).authorized, true);
  assert.deepEqual(args.permissionRequests, ["contributor", "maintainer"]);
});

test("the current label also authorizes subsequent PR commits", async () => {
  const args = fixture({ currentSha: "new-pr-sha", labels: ["ready-hcu"],
    events: [labeled("maintainer")], permissions: { maintainer: "write" } });
  args.context.payload.pull_request.head.sha = "new-pr-sha";
  await prepare(args);
  assert.equal(args.outputs.authorized, "true");
  assert.equal(args.statuses[0].sha, "new-pr-sha");
});

for (const events of [[], [labeled("reader")],
  [labeled("maintainer"), labeled("reader")],
  [labeled("maintainer"), labeled("maintainer", "unlabeled")]]) {
  test(`unverified or revoked label fails closed: ${JSON.stringify(events)}`, async () => {
    const args = fixture({ labels: ["ready-hcu"], events,
      permissions: { maintainer: "maintain" } });
    assert.equal((await authorize(args)).authorized, false);
  });
}

test("a removed label does not authorize using an older label event", async () => {
  const args = fixture({ events: [labeled("maintainer")],
    permissions: { maintainer: "maintain" } });
  assert.equal((await authorize(args)).authorized, false);
});

for (const overrides of [{ draft: true }, { state: "closed" }]) {
  test(`trusted contributors still need an open ready PR: ${JSON.stringify(overrides)}`, async () => {
    assert.equal((await authorize(fixture({ permission: "write", ...overrides }))).authorized, false);
  });
}

test("a stale event neither runs hardware nor overwrites a newer status", async () => {
  const args = fixture({ permission: "maintain", currentSha: "new-pr-sha" });
  await prepare(args);
  await finish(args, { authorizationResult: "success", testResult: "success" });
  assert.equal(args.outputs.authorized, "false");
  assert.deepEqual(args.statuses, []);
});

test("API failure cannot authorize hardware and is reported as an error", async () => {
  const args = fixture({ permission: "write", apiError: true });
  await assert.rejects(prepare(args), /GitHub API unavailable/);
  assert.equal(args.outputs.authorized, "false");
  await assert.rejects(finish(args, { authorizationResult: "failure", testResult: "skipped" }),
    /GitHub API unavailable/);
  assert.equal(args.statuses[0].state, "error");
});

test("publishing the pending status must succeed before granting authorization", async () => {
  const args = fixture({ permission: "write" });
  args.github.rest.repos.createCommitStatus = async () => { throw new Error("Forbidden"); };
  await assert.rejects(prepare(args), /Forbidden/);
  assert.equal(args.outputs.authorized, "false");
});

for (const testResult of ["success", "failure", "cancelled", "skipped"]) {
  test(`hardware outcome ${testResult} is reported on the PR head`, async () => {
    const args = fixture({ permission: "write" });
    await finish(args, { authorizationResult: "success", testResult });
    assert.equal(args.statuses[0].state, testResult === "success" ? "success" : "failure");
    assert.equal(args.statuses[0].sha, "pr-sha");
    assert.equal(args.failures.length, testResult === "success" ? 0 : 1);
  });
}

test("failed authorization cannot be turned into a passing check", async () => {
  const args = fixture({ permission: "write" });
  await finish(args, { authorizationResult: "failure", testResult: "success" });
  assert.equal(args.statuses[0].state, "failure");
});

test("authorization is rechecked before reporting successful hardware tests", async () => {
  const args = fixture(); // The ready-hcu label has since been removed.
  await finish(args, { authorizationResult: "success", testResult: "success" });
  assert.equal(args.statuses[0].state, "failure");
});

test("push, schedule and manual runs retain existing execution without PR status", async () => {
  const args = fixture();
  args.context.payload = {};
  await prepare(args);
  await finish(args, { authorizationResult: "success", testResult: "success" });
  assert.equal(args.outputs.authorized, "true");
  assert.equal(args.outputs.sha, "base-sha");
  assert.deepEqual(args.statuses, []);
  assert.deepEqual(args.permissionRequests, []);
});

test("documentation-only PR needs no hardware authorization and reports the skip", async () => {
  const args = fixture({ files: ["README.md", "docs/assets/flow.svg", "guide.rst",
    ".github/workflows/README.md", "LICENSE", "NOTICE"].map((filename) => ({ filename })) });
  await prepare(args);
  await finish(args, { authorizationResult: "success", testResult: "skipped" });
  assert.equal(args.outputs.authorized, "false");
  assert.deepEqual(args.permissionRequests, []);
  assert.equal(args.statuses.length, 2);
  for (const item of args.statuses) {
    assert.equal(item.state, "success");
    assert.match(item.description, /Documentation-only.*skipped/);
  }
  assert.deepEqual(args.failures, []);
});

for (const filename of ["turbo_physai/engine.py", "test/test_runner.py",
  "config.yaml", "requirements.txt", ".github/workflows/hcu-ci.yml", "scripts/check_docs.py"]) {
  test(`documentation mixed with ${filename} still runs tests`, async () => {
    const args = fixture({ permission: "maintain", files: [
      { filename: "docs/README.md" }, { filename },
    ] });
    await prepare(args);
    assert.equal(args.outputs.authorized, "true");
    assert.equal(args.statuses[0].state, "pending");
  });
}

test("renaming source code into a documentation path still needs tests", async () => {
  const args = fixture({ permission: "write", files: [
    { filename: "docs/example.md", previous_filename: "turbo_physai/engine.py", status: "renamed" },
  ] });
  assert.equal((await authorize(args)).authorized, true);
});

test("documentation deletion and documentation renames can skip tests", async () => {
  const args = fixture({ files: [
    { filename: "docs/old.md", status: "removed" },
    { filename: "docs/new.md", previous_filename: "README.md", status: "renamed" },
  ] });
  assert.equal((await authorize(args)).skipTests, true);
});

for (const options of [{ files: [] },
  { files: [{ filename: "README.md" }], changedFiles: 3001 }]) {
  test(`empty or incomplete file list cannot skip tests: ${JSON.stringify(options)}`, async () => {
    const args = fixture({ permission: "write", ...options });
    assert.equal((await authorize(args)).authorized, true);
  });
}

test("changed head during file listing cannot publish a documentation-only success", async () => {
  const args = fixture({ files: [{ filename: "README.md" }] });
  const get = args.github.rest.pulls.get;
  let calls = 0;
  args.github.rest.pulls.get = async (...params) => {
    const response = await get(...params);
    if (++calls > 1) response.data.head.sha = "new-pr-sha";
    return response;
  };
  await prepare(args);
  assert.equal(args.outputs.authorized, "false");
  assert.deepEqual(args.statuses, []);
});

test("failure to list changed files cannot skip tests or authorize execution", async () => {
  const args = fixture({ permission: "write" });
  args.github.paginate = async () => { throw new Error("Files API unavailable"); };
  await assert.rejects(prepare(args), /Files API unavailable/);
  assert.equal(args.outputs.authorized, "false");
});
