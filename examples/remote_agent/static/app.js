const state = {
  health: null,
  sessions: [],
  selectedSession: null,
  runs: [],
  selectedRun: null,
  runEvents: new Map(),
  liveEvents: [],
  activeRunId: null,
  pendingInput: "",
  running: false,
  error: "",
};

const els = {};

document.addEventListener("DOMContentLoaded", () => {
  bindElements();
  bindEvents();
  render();
  refreshAll();
});

function bindElements() {
  for (const id of [
    "healthStatus",
    "refreshAllButton",
    "newSessionButton",
    "selectedSessionTitle",
    "selectedSessionMeta",
    "refreshSessionButton",
    "archiveSessionButton",
    "runForm",
    "runInput",
    "streamToggle",
    "startRunButton",
    "messagesList",
    "threadScroll",
    "activeRunLabel",
  ]) {
    els[id] = document.getElementById(id);
  }
}

function bindEvents() {
  els.refreshAllButton.addEventListener("click", refreshAll);
  els.newSessionButton.addEventListener("click", createDefaultSession);
  els.refreshSessionButton.addEventListener("click", () => {
    if (state.selectedSession) selectSession(state.selectedSession.id);
  });
  els.archiveSessionButton.addEventListener("click", archiveSelectedSession);
  els.runForm.addEventListener("submit", startRun);
  els.runInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      els.runForm.requestSubmit();
    }
  });
}

async function refreshAll() {
  state.error = "";
  await Promise.allSettled([refreshHealth(), refreshSessions()]);
  if (!state.selectedSession && state.sessions.length === 0) {
    await createDefaultSession();
  }
  render();
}

async function refreshHealth() {
  try {
    state.health = await api("GET", "/healthz");
  } catch (error) {
    state.health = null;
    state.error = errorMessage(error);
  }
  renderHealth();
}

async function refreshSessions() {
  try {
    state.sessions = await api("GET", "/v1/sessions");
    if (!state.selectedSession && state.sessions.length > 0) {
      await selectSession(state.sessions[0].id);
    } else if (state.selectedSession) {
      const stillExists = state.sessions.some((session) => session.id === state.selectedSession.id);
      if (!stillExists) clearSelectedSession();
    }
  } catch (error) {
    state.error = errorMessage(error);
  }
}

async function createDefaultSession() {
  if (state.running) return;
  try {
    const result = await api("POST", "/v1/sessions", {
      metadata: {
        source: "web_app",
      },
    });
    state.sessions = await api("GET", "/v1/sessions");
    await selectSession(result.session_id);
  } catch (error) {
    state.error = errorMessage(error);
    render();
  }
}

async function selectSession(sessionId) {
  state.selectedSession = await api("GET", `/v1/sessions/${encodeURIComponent(sessionId)}`);
  state.selectedRun = null;
  state.runEvents = new Map();
  state.liveEvents = [];
  state.pendingInput = "";
  await refreshRuns();
  render();
}

async function archiveSelectedSession() {
  if (!state.selectedSession || state.running) return;
  try {
    await api("DELETE", `/v1/sessions/${encodeURIComponent(state.selectedSession.id)}`);
    clearSelectedSession();
    await refreshSessions();
  } catch (error) {
    state.error = errorMessage(error);
  }
  render();
}

function clearSelectedSession() {
  state.selectedSession = null;
  state.runs = [];
  state.selectedRun = null;
  state.runEvents = new Map();
  state.liveEvents = [];
  state.activeRunId = null;
  state.pendingInput = "";
}

async function refreshRuns() {
  if (!state.selectedSession) return;
  state.runs = await api(
    "GET",
    `/v1/sessions/${encodeURIComponent(state.selectedSession.id)}/runs`,
  );
  await refreshRunEvents();
  if (!state.selectedRun && state.runs.length > 0) state.selectedRun = state.runs[0];
}

async function selectRun(runId) {
  if (!state.selectedSession) return;
  state.selectedRun = await api(
    "GET",
    `/v1/sessions/${encodeURIComponent(state.selectedSession.id)}/runs/${encodeURIComponent(runId)}`,
  );
  await refreshRunEvents([state.selectedRun]);
  render();
}

async function refreshRunEvents(runs = state.runs) {
  if (!state.selectedSession) return;
  await Promise.all(
    runs.map(async (run) => {
      if (!run.id || state.runEvents.has(run.id)) return;
      const events = await api(
        "GET",
        `/v1/sessions/${encodeURIComponent(state.selectedSession.id)}/runs/${encodeURIComponent(run.id)}/events`,
      );
      state.runEvents.set(run.id, events);
    }),
  );
}

async function startRun(event) {
  event.preventDefault();
  if (state.running) return;
  if (!state.selectedSession) {
    await createDefaultSession();
  }
  if (!state.selectedSession) return;

  const input = els.runInput.value.trim();
  if (!input) return;

  state.liveEvents = [];
  state.selectedRun = {
    input,
    status: "running",
  };
  state.pendingInput = input;
  state.activeRunId = null;
  state.running = true;
  state.error = "";
  render();

  const body = {
    input,
    stream: els.streamToggle.checked,
  };

  try {
    if (body.stream) {
      await streamRun(body);
    } else {
      const run = await api(
        "POST",
        `/v1/sessions/${encodeURIComponent(state.selectedSession.id)}/runs`,
        body,
      );
      state.selectedRun = run;
      state.activeRunId = run.id;
      await afterRunFinished(run.id);
    }
    els.runInput.value = "";
  } catch (error) {
    state.error = errorMessage(error);
  } finally {
    state.running = false;
    render();
  }
}

async function streamRun(body) {
  const path = `/v1/sessions/${encodeURIComponent(state.selectedSession.id)}/runs`;
  const response = await fetch(path, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      accept: "text/event-stream",
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }
  if (!response.body) {
    throw new Error("stream response body is empty");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() || "";
    for (const frame of frames) {
      handleSseFrame(frame);
    }
  }

  if (buffer.trim()) handleSseFrame(buffer);
  if (state.activeRunId) {
    await afterRunFinished(state.activeRunId);
  } else {
    await refreshRuns();
  }
}

function handleSseFrame(frame) {
  const parsed = parseSseFrame(frame);
  if (!parsed) return;

  if (parsed.event === "run_accepted") {
    state.activeRunId = parsed.data.run_id;
    state.selectedRun = {
      id: parsed.data.run_id,
      session_id: parsed.data.session_id,
      status: "running",
      input: state.pendingInput,
    };
  } else if (parsed.event === "error") {
    state.error = parsed.data.message || JSON.stringify(parsed.data);
  } else {
    state.liveEvents.push({
      name: parsed.event,
      data: parsed.data,
      at: Date.now(),
    });
  }

  render();
}

async function afterRunFinished(runId) {
  await refreshRuns();
  await selectRun(runId);
  state.liveEvents = [];
  state.pendingInput = "";
}

async function api(method, path, body) {
  const options = { method, headers: {} };
  if (body !== null && body !== undefined) {
    options.headers["content-type"] = "application/json";
    options.body = JSON.stringify(body);
  }

  const response = await fetch(path, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `${method} ${path} failed`);
  }
  if (response.status === 204) return null;
  return response.json();
}

function parseSseFrame(frame) {
  let event = "message";
  const data = [];
  for (const line of frame.split(/\r?\n/)) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }
  if (data.length === 0) return null;
  const raw = data.join("\n");
  try {
    return { event, data: JSON.parse(raw) };
  } catch {
    return { event, data: raw };
  }
}

function render() {
  renderHealth();
  renderSelectedSession();
  renderControls();
  renderMessages();
}

function renderHealth() {
  const dot = state.health ? "status-ok" : state.error ? "status-error" : "status-muted";
  const text = state.health ? `ok · tools ${state.health.tool_count}` : state.error ? "error" : "healthz";
  els.healthStatus.innerHTML = `<span class="status-dot ${dot}"></span><span>${escapeHtml(text)}</span>`;
}

function renderSelectedSession() {
  if (!state.selectedSession) {
    els.selectedSessionTitle.textContent = "准备 session";
    els.selectedSessionMeta.textContent = state.error || "正在连接 Gateway";
    return;
  }
  els.selectedSessionTitle.textContent = shorten(state.selectedSession.id, 42);
  els.selectedSessionMeta.textContent = `${state.selectedSession.model} · ${state.selectedSession.status}`;
}

function renderControls() {
  const hasSession = Boolean(state.selectedSession);
  const visibleRunId = state.activeRunId || (state.selectedRun && state.selectedRun.id);
  els.startRunButton.disabled = !hasSession || state.running;
  els.runInput.disabled = !hasSession || state.running;
  els.archiveSessionButton.disabled = !hasSession || state.running;
  els.refreshSessionButton.disabled = !hasSession;
  els.activeRunLabel.textContent = state.running
    ? visibleRunId || "running"
    : visibleRunId || "idle";
}

function renderMessages() {
  const items = conversationItems();
  if (items.length === 0) {
    els.messagesList.innerHTML = `<div class="empty">发送一条任务开始。</div>`;
    return;
  }

  els.messagesList.innerHTML = items.map(renderMessage).join("");
  if (state.running) {
    els.threadScroll.scrollTop = els.threadScroll.scrollHeight;
  }
}

function conversationItems() {
  const items = [];
  const runs = [...state.runs].sort((a, b) => (a.started_at || 0) - (b.started_at || 0));

  for (const run of runs) {
    items.push(...itemsForRun(run, eventsForRun(run)));
  }

  if (state.running && state.selectedRun && !state.selectedRun.id) {
    items.push(...itemsForRun(state.selectedRun, state.liveEvents));
  } else if (state.running && state.activeRunId && !state.runs.some((run) => run.id === state.activeRunId)) {
    items.push(...itemsForRun(state.selectedRun, state.liveEvents));
  }

  if (state.running && !items.some((item) => item.pending)) {
    items.push({
      kind: "thinking",
      role: "Agent 思考",
      content: "正在思考",
      pending: true,
    });
  }

  if (state.error) {
    items.push({
      kind: "error",
      role: "错误",
      content: state.error,
    });
  }

  return items;
}

function itemsForRun(run, events) {
  const items = [];
  const input = (run && run.input) || state.pendingInput;
  if (input) {
    items.push({
      kind: "user",
      role: "user",
      content: input,
    });
  }
  items.push(...displayItemsFromEvents(events));
  return items;
}

function eventsForRun(run) {
  if (!run || !run.id) return state.liveEvents;
  if (run.id === state.activeRunId && state.liveEvents.length > 0) return state.liveEvents;
  return (state.runEvents.get(run.id) || []).map((event) => ({
    name: event.event_type,
    data: event.payload,
    at: event.ts,
  }));
}

function displayItemsFromEvents(events) {
  const stepFinishSteps = new Set();
  for (const event of events) {
    const variant = variantOf(event.data);
    if (variant.name === "StepFinish") {
      const step = variant.value.event;
      if (step && step.step) stepFinishSteps.add(step.step);
    }
  }

  const items = [];
  const partialTextByStep = new Map();
  const toolStepCandidates = new Set();
  const liveToolItems = new Map();
  let sawDone = false;

  for (const event of events) {
    const variant = variantOf(event.data);
    if (variant.name === "Model") {
      collectModelDelta(variant.value, partialTextByStep, toolStepCandidates);
      continue;
    }
    if (variant.name === "StepFinish") {
      items.push(...itemsFromStep(variant.value.event));
    } else if (variant.name === "ToolCallStart") {
      const start = variant.value.event;
      if (!stepFinishSteps.has(start.step)) {
        pushPartialStepItem(items, partialTextByStep, toolStepCandidates, start.step);
        const item = toolUseItem(start.tool_call, null, start.step);
        liveToolItems.set(start.tool_call.call_id, item);
        items.push(item);
      }
    } else if (variant.name === "ToolCallFinish") {
      const finish = variant.value.event;
      if (!stepFinishSteps.has(finish.step)) {
        const item = liveToolItems.get(finish.tool_call.call_id);
        if (item) {
          item.result = finish.tool_result;
          item.content = toolUseContent(finish.tool_call, finish.tool_result);
        } else {
          items.push(toolUseItem(finish.tool_call, finish.tool_result, finish.step));
        }
      }
    } else if (variant.name === "Done") {
      sawDone = true;
    }
  }

  for (const [step] of partialTextByStep) {
    if (!stepFinishSteps.has(step)) {
      pushPartialStepItem(items, partialTextByStep, toolStepCandidates, step);
    }
  }

  if (items.length === 0 && sawDone) {
    const done = events
      .map((event) => variantOf(event.data))
      .find((variant) => variant.name === "Done");
    const output = done && done.value.output && done.value.output.output_text;
    if (output) {
      items.push({
        kind: "final",
        role: "结果",
        content: output,
      });
    }
  }

  return items;
}

function collectModelDelta(model, partialTextByStep, toolStepCandidates) {
  if (!model || !model.event) return;
  const streamEvent = variantOf(model.event);
  if (streamEvent.name === "TextDelta") {
    const text = streamEvent.value.text || "";
    if (!text) return;
    partialTextByStep.set(model.step, `${partialTextByStep.get(model.step) || ""}${text}`);
  } else if (streamEvent.name === "ToolCallReady") {
    toolStepCandidates.add(model.step);
  }
}

function pushPartialStepItem(items, partialTextByStep, toolStepCandidates, step) {
  const text = (partialTextByStep.get(step) || "").trim();
  if (!text) return;
  items.push({
    kind: toolStepCandidates.has(step) ? "thinking" : "final",
    role: toolStepCandidates.has(step) ? "Agent 思考" : "结果",
    meta: `step ${step}`,
    content: text,
    pending: true,
  });
  partialTextByStep.delete(step);
}

function itemsFromStep(step) {
  if (!step) return [];
  const items = [];
  const toolCalls = step.tool_calls || [];
  const toolResults = step.tool_results || [];
  const thought = stepThought(step);

  if (toolCalls.length > 0) {
    items.push({
      kind: "thinking",
      role: "Agent 思考",
      meta: `step ${step.step}`,
      content: thought || "决定调用工具。",
    });
    for (const call of toolCalls) {
      const result = toolResults.find((candidate) => candidate.call_id === call.call_id);
      items.push(toolUseItem(call, result, step.step));
    }
  } else if (step.output_text) {
    items.push({
      kind: "final",
      role: "结果",
      meta: `step ${step.step}`,
      content: step.output_text,
    });
  }

  return items;
}

function stepThought(step) {
  if (step.reasoning_text && step.reasoning_text.trim()) return step.reasoning_text.trim();
  const reasoning = (step.reasoning_parts || [])
    .map((part) => part.text || "")
    .join("")
    .trim();
  if (reasoning) return reasoning;
  if (step.output_text && step.output_text.trim()) return step.output_text.trim();
  return "";
}

function toolUseItem(call, result, step) {
  return {
    kind: "tool",
    role: "工具调用",
    meta: `${call.tool_name} · step ${step}`,
    content: toolUseContent(call, result),
    result,
  };
}

function toolUseContent(call, result) {
  return [
    "参数",
    prettyJson(call.args_json || {}),
    "",
    "结果",
    result ? prettyJson(result.output_json) : "等待工具返回结果。",
  ].join("\n");
}

function renderMessage(item) {
  const meta = item.meta ? `<span class="message-meta">${escapeHtml(item.meta)}</span>` : "";
  const pending = item.pending ? " thinking-pulse" : "";
  if (item.kind === "tool") {
    const [args, result] = splitToolContent(item.content);
    return `
      <article class="message-row tool">
        <div class="message-role">${escapeHtml(item.role)}</div>
        <details class="tool-card">
          <summary>
            <span>
              ${meta}
              <span class="tool-title">参数</span>
              <code>${escapeHtml(args)}</code>
            </span>
          </summary>
          <pre>${escapeHtml(result)}</pre>
        </details>
      </article>
    `;
  }
  return `
    <article class="message-row ${escapeAttr(item.kind)}">
      <div class="message-role">${escapeHtml(item.role)}</div>
      <div class="message-content${pending}">${meta}${escapeHtml(item.content)}</div>
    </article>
  `;
}

function splitToolContent(content) {
  const marker = "\n\n结果\n";
  const index = content.indexOf(marker);
  if (index === -1) return [content.replace(/^参数\n/, ""), ""];
  return [
    content.slice(0, index).replace(/^参数\n/, ""),
    content.slice(index + marker.length),
  ];
}

function prettyJson(value) {
  if (value === undefined || value === null) return "null";
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}

function errorMessage(error) {
  return String(error && error.message ? error.message : error);
}

function variantOf(value) {
  if (!value || typeof value !== "object") return { name: "", value: null };
  const keys = Object.keys(value);
  if (keys.length === 0) return { name: "", value: null };
  return { name: keys[0], value: value[keys[0]] };
}

function shorten(value, max) {
  const text = String(value || "");
  if (text.length <= max) return text;
  return `${text.slice(0, Math.max(0, max - 1))}…`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeAttr(value) {
  return escapeHtml(value);
}
