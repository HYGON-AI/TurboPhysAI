// Copyright 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

const assert = require("node:assert/strict");
const { test } = require("node:test");
const { authorize, prepare, finish } = require("../hcu_authorization.cjs");
// Match the REST API: triage maps to read, maintain maps to write.
const ROLES = {
  admin: { permission: "admin", role_name: "admin" },
  maintain: { permission: "write", role_name: "maintain" },
  write: { permission: "write", role_name: "write" },
  triage: { permission: "read", role_name: "triage" },
  read: { permission: "read", role_name: "read" },
  none: { permission: "none", role_name: "none" },
};

function fixture({ role = "read", labels = [], events = [], draft = false,
  state = "open", currentSha = "pr-sha", currentBase = "base-sha", roles = {},
  files = [{ filename: "turbo_physai/__init__.py" }], changedFiles = files.length } = {}) {
  const outputs = {};
  const failures = [];
  const messages = [];
  const permissionRequests = [];
  const context = {
    eventName: "pull_request",
    repo: { owner: "example", repo: "project" },
    sha: "merge-sha",
    payload: { action: "opened", pull_request: {
      number: 9, head: { sha: "pr-sha", repo: { full_name: "contributor/project" } },
      base: { sha: "base-sha", repo: { full_name: "example/project" } },
    } },
  };
  const pr = {
    number: 9, state, draft, head: { sha: currentSha }, base: { sha: currentBase },
    changed_files: changedFiles, user: { login: "contributor" },
    labels: labels.map((name) => ({ name })),
  };
  const github = {
    rest: {
      pulls: { get: async () => ({ data: structuredClone(pr) }), listFiles: Symbol("listFiles") },
      repos: {
        getCollaboratorPermissionLevel: async ({ username }) => {
          permissionRequests.push(username);
          const value = roles[username] ?? role;
          return { data: structuredClone(typeof value === "string" ?
            ROLES[value] ?? { permission: value, role_name: value } : value) };
        },
        createCommitStatus: async () => { throw new Error("Read-only token: status writes forbidden"); },
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
    info: (message) => messages.push(message),
    setFailed: (message) => failures.push(message),
  };
  return { github, context, core, outputs, failures, messages, permissionRequests, pr };
}

const labeled = (actor, event = "labeled") => ({
  event, actor: { login: actor }, label: { name: "ready-hcu" },
});

for (const role of ["admin", "maintain", "write", "triage"]) {
  test(`${role} is authorized with the real REST permission/role combination`, async () => {
    const args = fixture({ role });
    await prepare(args);
    assert.equal(args.outputs.authorized, "true");
    assert.equal(args.outputs.sha, "merge-sha");
    await finish(args, { authorizationResult: "success", testResult: "success" });
    assert.deepEqual(args.failures, []);
  });
  test(`${role} can authorize an external contributor with ready-hcu`, async () => {
    const args = fixture({ labels: ["ready-hcu"], events: [labeled("maintainer")],
      roles: { maintainer: role } });
    assert.equal((await authorize(args)).authorized, true);
    assert.deepEqual(args.permissionRequests, ["contributor", "maintainer"]);
  });
}

for (const role of ["read", "none", "unknown", {},
  { permission: "read", role_name: "custom-reader" },
  { permission: "write", role_name: "custom-writer" }]) {
  test(`unknown/read roles cannot gain hardware access: ${JSON.stringify(role)}`, async () => {
    const args = fixture({ role });
    await prepare(args);
    assert.equal(args.outputs.authorized, String(role.permission === "write"));
  });
}

test("organization membership alone does not grant hardware access", async () => {
  const args = fixture();
  args.pr.author_association = "MEMBER";
  assert.equal((await authorize(args)).authorized, false);
});

test("the current label also authorizes subsequent PR commits", async () => {
  const args = fixture({ currentSha: "new-pr-sha", labels: ["ready-hcu"],
    events: [labeled("maintainer")], roles: { maintainer: "write" } });
  args.context.payload.pull_request.head.sha = "new-pr-sha";
  args.context.sha = "new-merge-sha";
  await prepare(args);
  assert.equal(args.outputs.authorized, "true");
  assert.equal(args.outputs.sha, "new-merge-sha");
});

for (const events of [[], [labeled("reader")],
  [labeled("maintainer"), labeled("reader")],
  [labeled("maintainer"), labeled("maintainer", "unlabeled")]]) {
  test(`unverified or revoked label fails closed: ${JSON.stringify(events)}`, async () => {
    const args = fixture({ labels: ["ready-hcu"], events, roles: { maintainer: "maintain" } });
    assert.equal((await authorize(args)).authorized, false);
  });
}

test("a removed label cannot authorize using historical events", async () => {
  const args = fixture({ events: [labeled("maintainer")], roles: { maintainer: "maintain" } });
  assert.equal((await authorize(args)).authorized, false);
});

test("label removed during permission lookup fails closed", async () => {
  const args = fixture({ labels: ["ready-hcu"], events: [labeled("maintainer")],
    roles: { maintainer: "write" } });
  const lookup = args.github.rest.repos.getCollaboratorPermissionLevel;
  args.github.rest.repos.getCollaboratorPermissionLevel = async (query) => {
    if (query.username === "maintainer") args.pr.labels = [];
    return lookup(query);
  };
  assert.equal((await authorize(args)).authorized, false);
});

for (const options of [{ draft: true }, { state: "closed" },
  { currentSha: "new-pr-sha" }]) {
  test(`ineligible PR does not get a passing result: ${JSON.stringify(options)}`, async () => {
    const args = fixture({ role: "write", ...options });
    await prepare(args);
    assert.equal(args.outputs.authorized, "false");
    await finish(args, { authorizationResult: "success", testResult: "success" });
    assert.equal(args.failures.length, 1);
  });
}

for (const endpoint of ["permission", "files", "pull", "events"]) {
  test(`${endpoint} API error cannot authorize hardware`, async () => {
    const args = fixture({ labels: ["ready-hcu"] });
    const fail = async () => { throw new Error("GitHub API unavailable"); };
    if (endpoint === "permission") args.github.rest.repos.getCollaboratorPermissionLevel = fail;
    if (endpoint === "pull") args.github.rest.pulls.get = fail;
    if (["files", "events"].includes(endpoint)) {
      const paginate = args.github.paginate;
      args.github.paginate = (method, options) =>
        method === args.github.rest[endpoint === "files" ? "pulls" : "issues"]
          [endpoint === "files" ? "listFiles" : "listEvents"] ? fail() : paginate(method, options);
    }
    await assert.rejects(prepare(args), /GitHub API unavailable/);
    assert.equal(args.outputs.authorized, "false");
    await assert.rejects(finish(args, { authorizationResult: "success", testResult: "success" }),
      /GitHub API unavailable/);
  });
}

for (const authorizationResult of ["success", "failure", "cancelled", "skipped"]) {
  for (const testResult of ["success", "failure", "cancelled", "skipped"]) {
    test(`gate: authorization=${authorizationResult}, hardware=${testResult}`, async () => {
      const args = fixture({ role: "write" });
      await finish(args, { authorizationResult, testResult });
      const passed = authorizationResult === "success" && testResult === "success";
      assert.equal(args.failures.length, passed ? 0 : 1);
    });
  }
}

test("withdrawn authorization cannot produce a passing gate", async () => {
  const args = fixture({ labels: ["ready-hcu"], events: [labeled("maintainer")],
    roles: { maintainer: "write" } });
  await prepare(args);
  assert.equal(args.outputs.authorized, "true");
  args.pr.labels = [];
  await finish(args, { authorizationResult: "success", testResult: "success" });
  assert.equal(args.failures.length, 1);
});

for (const eventName of ["push", "schedule", "workflow_dispatch"]) {
  test(`${eventName} tests the selected revision without PR API calls`, async () => {
    const args = fixture();
    args.context.eventName = eventName;
    args.context.payload = {};
    args.context.sha = "repository-sha";
    await prepare(args);
    assert.equal(args.outputs.authorized, "true");
    assert.equal(args.outputs.sha, "repository-sha");
    await finish(args, { authorizationResult: "success", testResult: "failure" });
    assert.equal(args.failures.length, 1);
    assert.deepEqual(args.permissionRequests, []);
  });
}

for (const eventName of ["pull_request", "pull_request_target", "issue_comment"]) {
  test(`missing or unsupported ${eventName} payload is rejected`, async () => {
    const args = fixture();
    args.context.eventName = eventName;
    args.context.payload = {};
    await assert.rejects(prepare(args), /Expected a pull_request/);
    assert.equal(args.outputs.authorized, "false");
  });
}

test("documentation-only PR skips hardware without permission lookups", async () => {
  const args = fixture({ files: ["README.md", "docs/assets/flow.svg", "guide.rst",
    ".github/workflows/README.md", "LICENSE", "NOTICE"].map((filename) => ({ filename })) });
  await prepare(args);
  await finish(args, { authorizationResult: "success", testResult: "skipped" });
  assert.equal(args.outputs.authorized, "false");
  assert.deepEqual(args.permissionRequests, []);
  assert.deepEqual(args.failures, []);
  assert.match(args.messages.at(-1), /Documentation-only.*skipped/);
});

for (const filename of ["turbo_physai/engine.py", "test/test_runner.py",
  "config.yaml", "requirements.txt", ".github/workflows/hcu-ci.yml", "scripts/check_docs.py"]) {
  test(`documentation mixed with ${filename} needs hardware`, async () => {
    const args = fixture({ role: "maintain", files: [{ filename: "docs/README.md" }, { filename }] });
    assert.equal((await authorize(args)).authorized, true);
  });
}

test("renaming source code into documentation still needs tests", async () => {
  const args = fixture({ role: "write", files: [
    { filename: "docs/example.md", previous_filename: "turbo_physai/engine.py", status: "renamed" },
  ] });
  assert.equal((await authorize(args)).authorized, true);
});

test("documentation deletion and renaming can skip tests", async () => {
  const args = fixture({ files: [
    { filename: "docs/old.md", status: "removed" },
    { filename: "docs/new.md", previous_filename: "README.md", status: "renamed" },
  ] });
  assert.equal((await authorize(args)).skipTests, true);
});

for (const options of [{ files: [] }, { files: [{ filename: "README.md" }], changedFiles: 3001 }]) {
  test(`empty or incomplete file list cannot skip tests: ${JSON.stringify(options)}`, async () => {
    assert.equal((await authorize(fixture({ role: "write", ...options }))).authorized, true);
  });
}

for (const update of [{ head: { sha: "new-pr-sha" } },
  { draft: true }, { state: "closed" }]) {
  test(`PR changing during file listing cannot skip tests: ${JSON.stringify(update)}`, async () => {
    const args = fixture({ files: [{ filename: "README.md" }] });
    const paginate = args.github.paginate;
    args.github.paginate = (...params) => {
      Object.assign(args.pr, update);
      return paginate(...params);
    };
    assert.equal((await authorize(args)).skipTests, undefined);
  });
}

test("base metadata changing does not invalidate the pinned merge revision", async () => {
  const args = fixture({ role: "write", currentBase: "new-base-sha" });
  await prepare(args);
  assert.equal(args.outputs.authorized, "true");
  assert.equal(args.outputs.sha, "merge-sha");
});
