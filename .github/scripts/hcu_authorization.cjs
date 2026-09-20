// Copyright 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

const TRUSTED_PERMISSIONS = new Set(["admin", "maintain", "write", "triage"]);

function isDocumentation(path) {
  return typeof path === "string" &&
    (path.startsWith("docs/") || /\.(md|rst)$/.test(path) ||
      ["LICENSE", "NOTICE"].includes(path));
}

async function authorize({ github, context }) {
  const eventPr = context.payload.pull_request;
  if (["push", "schedule", "workflow_dispatch"].includes(context.eventName) && !eventPr) {
    return { authorized: true, sha: context.sha, reason: "Repository workflow" };
  }
  if (context.eventName !== "pull_request" || !eventPr?.head?.sha) {
    throw new Error("Expected a pull_request event with a head revision");
  }

  const { data: pr } = await github.rest.pulls.get({
    ...context.repo,
    pull_number: eventPr.number,
  });
  // pull_request's SHA identifies the tested merge, not the fork head.
  const sha = context.sha;
  function ineligible(current) {
    if (current.head.sha !== eventPr.head.sha) {
      return { authorized: false, sha, stale: true, reason: "PR head has changed" };
    }
    if (current.state !== "open" || current.draft) {
      return { authorized: false, sha, reason: "PR must be open and ready for review" };
    }
    return null;
  }
  const rejected = ineligible(pr);
  if (rejected) return rejected;

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
  const changed = ineligible(currentPr);
  if (changed) return changed;
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
    // GitHub reports triage as permission=read, role_name=triage.
    return TRUSTED_PERMISSIONS.has(data.permission) ||
      TRUSTED_PERMISSIONS.has(data.role_name);
  }

  if (await trusted(pr.user.login)) {
    return { authorized: true, sha, reason: `Auto-authorized for ${pr.user.login}` };
  }
  if (!currentPr.labels.some((label) => label.name === "ready-hcu")) {
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
  const { data: latestPr } = await github.rest.pulls.get({
    ...context.repo, pull_number: pr.number,
  });
  const latestRejection = ineligible(latestPr);
  if (latestRejection) return latestRejection;
  if (!latestPr.labels.some((label) => label.name === "ready-hcu")) {
    return { authorized: false, sha, reason: "ready-hcu has been removed" };
  }
  return { authorized: true, sha, reason: `ready-hcu authorized by ${actor}` };
}

async function prepare(args) {
  const { core } = args;
  // Fail closed: the HCU job requires this output to be explicitly true.
  core.setOutput("authorized", "false");
  const decision = await authorize(args);
  core.setOutput("sha", decision.sha);
  core.info(decision.reason);
  core.setOutput("authorized", String(decision.authorized));
}

async function finish(args, { authorizationResult, testResult }) {
  const { core } = args;
  // Native Actions job results work with a fork's read-only token.
  // Recheck authorization so withdrawn permission cannot produce a green gate.
  const decision = await authorize(args);
  const passed = authorizationResult === "success" && !decision.stale &&
    (decision.skipTests ? testResult === "skipped" :
      decision.authorized && testResult === "success");
  const description = authorizationResult !== "success" ? "HCU authorization job did not succeed" :
    decision.skipTests || !decision.authorized ? decision.reason :
    passed ? "HCU build and tests passed" : "HCU build or tests did not complete successfully";
  core.info(description);
  if (!passed) core.setFailed(description);
}

module.exports = { authorize, prepare, finish };
