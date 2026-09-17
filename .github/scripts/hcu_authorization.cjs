// Copyright 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

const TRUSTED_PERMISSIONS = new Set(["admin", "maintain", "write", "triage"]);
const STATUS_CONTEXT = "HCU CI";

function isDocumentation(path) {
  return typeof path === "string" &&
    (path.startsWith("docs/") || /\.(md|rst)$/.test(path) ||
      ["LICENSE", "NOTICE"].includes(path));
}

async function authorize({ github, context }) {
  const eventPr = context.payload.pull_request;
  if (!eventPr) {
    return { authorized: true, sha: context.sha, reason: "Repository workflow" };
  }

  const { data: pr } = await github.rest.pulls.get({
    ...context.repo,
    pull_number: eventPr.number,
  });
  const sha = eventPr.head.sha;
  if (pr.head.sha !== sha) {
    return { authorized: false, sha, stale: true, reason: "PR head has changed" };
  }
  if (pr.state !== "open" || pr.draft) {
    return { authorized: false, sha, reason: "PR must be open and ready for review" };
  }

  const files = await github.paginate(github.rest.pulls.listFiles, {
    ...context.repo,
    pull_number: pr.number,
    per_page: 100,
  });
  // The files endpoint is mutable and capped at 3,000 files. Never skip tests
  // based on an incomplete list or a PR that changed while listing its files.
  const { data: currentPr } = await github.rest.pulls.get({
    ...context.repo,
    pull_number: pr.number,
  });
  if (currentPr.head.sha !== sha) {
    return { authorized: false, sha, stale: true, reason: "PR head has changed" };
  }
  if (files.length > 0 && files.length === pr.changed_files &&
    files.every((file) => isDocumentation(file.filename) &&
      (!file.previous_filename || isDocumentation(file.previous_filename)))) {
    return { authorized: false, sha, skipTests: true,
      reason: "Documentation-only change; HCU build and tests skipped" };
  }

  async function trusted(username) {
    const { data } = await github.rest.repos.getCollaboratorPermissionLevel({
      ...context.repo,
      username,
    });
    return TRUSTED_PERMISSIONS.has(data.permission);
  }

  if (await trusted(pr.user.login)) {
    return { authorized: true, sha, reason: `Auto-authorized for ${pr.user.login}` };
  }
  if (!pr.labels.some((label) => label.name === "ready-hcu")) {
    return { authorized: false, sha, reason: "Waiting for a trusted ready-hcu label" };
  }

  const events = await github.paginate(github.rest.issues.listEvents, {
    ...context.repo,
    issue_number: pr.number,
    per_page: 100,
  });
  const labelEvent = events
    .filter((event) => ["labeled", "unlabeled"].includes(event.event) &&
      event.label?.name === "ready-hcu")
    .at(-1);
  const actor = labelEvent?.actor?.login;
  if (labelEvent?.event !== "labeled" || !actor || !(await trusted(actor))) {
    return { authorized: false, sha, reason: "ready-hcu requires a trusted label author" };
  }
  return { authorized: true, sha, reason: `ready-hcu authorized by ${actor}` };
}

async function status({ github, context }, sha, state, description) {
  await github.rest.repos.createCommitStatus({
    ...context.repo,
    sha,
    state,
    context: STATUS_CONTEXT,
    description,
    target_url: `${context.serverUrl}/${context.repo.owner}/${context.repo.repo}/actions/runs/${context.runId}`,
  });
}

async function prepare(args) {
  const { core, context } = args;
  // Fail closed: the HCU job requires this output to be explicitly true.
  core.setOutput("authorized", "false");
  const decision = await authorize(args);
  core.setOutput("sha", decision.sha);
  core.info(decision.reason);
  if (context.payload.pull_request && !decision.stale) {
    await status(args, decision.sha,
      decision.skipTests ? "success" : decision.authorized ? "pending" : "failure",
      decision.authorized ? "HCU tests queued or running" : decision.reason);
  }
  core.setOutput("authorized", String(decision.authorized));
}

async function finish(args, { authorizationResult, testResult }) {
  const { core, context } = args;
  if (!context.payload.pull_request) return;
  let decision;
  try {
    // Recheck current head and authorization before publishing a passing result.
    decision = await authorize(args);
  } catch (error) {
    await status(args, context.payload.pull_request.head.sha, "error",
      "Unable to verify HCU authorization; see workflow logs");
    throw error;
  }
  if (decision.stale) {
    core.info(decision.reason);
    return;
  }
  const passed = authorizationResult === "success" &&
    (decision.skipTests ? testResult === "skipped" :
      decision.authorized && testResult === "success");
  const description = decision.skipTests || !decision.authorized ? decision.reason :
    passed ? "HCU build and tests passed" : "HCU build or tests did not complete successfully";
  await status(args, decision.sha, passed ? "success" : "failure", description);
  if (!passed) core.setFailed(description);
}

module.exports = { authorize, prepare, finish };
