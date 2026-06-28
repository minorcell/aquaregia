const STORAGE_KEY = "aquaregia.chatgpt.threads.v1";

const sidebarToggleEl = document.querySelector("#sidebar-toggle");
const threadsEl = document.querySelector("#threads");
const messagesEl = document.querySelector("#messages");
const formEl = document.querySelector("#composer");
const promptEl = document.querySelector("#prompt");
const sendEl = document.querySelector("#send");
const stopEl = document.querySelector("#stop");
const clearCurrentEl = document.querySelector("#clear-current");
const clearAllEl = document.querySelector("#clear-all");
const newChatEl = document.querySelector("#new-chat");
const chatNameEl = document.querySelector("#chat-name");
const turnStatusEl = document.querySelector("#turn-status");

let threads = loadThreads();
let activeThreadId = threads[0]?.id ?? createThread().id;
let activeAbort = null;

render();
promptEl.focus();

newChatEl.addEventListener("click", () => {
  activeThreadId = createThread().id;
  closeSidebarOnSmallScreen();
  render();
  promptEl.focus();
});

clearCurrentEl.addEventListener("click", () => {
  const thread = activeThread();
  thread.messages = [];
  thread.title = "New chat";
  thread.updatedAt = Date.now();
  saveThreads();
  render();
  promptEl.focus();
});

clearAllEl.addEventListener("click", () => {
  if (activeAbort) activeAbort.abort();
  threads = [newThread()];
  activeThreadId = threads[0].id;
  saveThreads();
  setBusy(false);
  render();
  promptEl.focus();
});

sidebarToggleEl.addEventListener("click", () => {
  document.body.classList.toggle("sidebar-open");
});

stopEl.addEventListener("click", () => {
  if (activeAbort) activeAbort.abort();
});

formEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  await submitPrompt();
});

promptEl.addEventListener("input", () => {
  autosizePrompt();
});

promptEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    formEl.requestSubmit();
  }
});

async function submitPrompt(value = promptEl.value) {
  const content = value.trim();
  if (!content || activeAbort) return;

  const thread = activeThread();
  const startedThreadId = thread.id;
  promptEl.value = "";
  autosizePrompt();
  closeSidebarOnSmallScreen();

  thread.messages.push({ role: "user", content });
  if (thread.title === "New chat") {
    thread.title = titleFrom(content);
  }

  const assistant = { role: "assistant", content: "", state: "streaming" };
  thread.messages.push(assistant);
  thread.updatedAt = Date.now();
  saveThreads();
  render();
  setBusy(true);

  const controller = new AbortController();
  activeAbort = controller;

  try {
    const payload = {
      messages: thread.messages
        .filter((message) => message.role === "user" || message.role === "assistant")
        .filter((message) => message.content.trim().length > 0)
        .map(({ role, content }) => ({ role, content })),
    };

    const response = await fetch("/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    if (!response.ok || !response.body) {
      throw new Error(await response.text());
    }

    for await (const event of readSse(response.body)) {
      if (event.name === "text_delta") {
        assistant.content += event.data;
        if (activeThreadId === startedThreadId) {
          updateStreamingMessage(assistant.content);
        }
      } else if (event.name === "usage") {
        if (activeThreadId === startedThreadId) {
          turnStatusEl.textContent = usageLabel(event.data);
        }
      } else if (event.name === "error") {
        throw new Error(event.data);
      }
    }

    assistant.state = "done";
    thread.updatedAt = Date.now();
    saveThreads();
  } catch (error) {
    if (error.name === "AbortError") {
      assistant.state = "stopped";
      if (!assistant.content) assistant.content = "Stopped.";
    } else {
      assistant.state = "error";
      assistant.content = error instanceof Error ? error.message : String(error);
    }
  } finally {
    activeAbort = null;
    setBusy(false);
    saveThreads();
    if (activeThreadId === startedThreadId) {
      render();
    } else {
      renderThreads();
    }
    promptEl.focus();
  }
}

function render() {
  renderThreads();
  renderMessages();
  const thread = activeThread();
  chatNameEl.textContent = thread.title;
  turnStatusEl.textContent = activeAbort ? "Generating" : "Ready";
}

function renderThreads() {
  threadsEl.replaceChildren();

  for (const thread of threads) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "thread";
    button.dataset.active = String(thread.id === activeThreadId);
    button.innerHTML = `
      <span class="thread-title"></span>
      <span class="thread-meta"></span>
    `;
    button.querySelector(".thread-title").textContent = thread.title;
    button.querySelector(".thread-meta").textContent = threadSubtitle(thread);
    button.addEventListener("click", () => {
      activeThreadId = thread.id;
      closeSidebarOnSmallScreen();
      render();
      promptEl.focus();
    });
    threadsEl.append(button);
  }
}

function renderMessages() {
  const thread = activeThread();
  messagesEl.replaceChildren();

  if (thread.messages.length === 0) {
    messagesEl.append(emptyState());
    return;
  }

  for (const message of thread.messages) {
    messagesEl.append(messageNode(message));
  }

  scrollToBottom();
}

function emptyState() {
  const section = document.createElement("section");
  section.className = "empty-state";

  const title = document.createElement("h2");
  title.textContent = "What are we working on?";

  const subtitle = document.createElement("p");
  subtitle.textContent = "Pick a starting point or write your own prompt.";

  const suggestions = document.createElement("div");
  suggestions.className = "suggestions";

  for (const prompt of [
    "Summarize this repository",
    "Explain the agent API",
    "Draft a minimal Rust example",
    "Review the current design",
  ]) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "suggestion";
    button.textContent = prompt;
    button.addEventListener("click", () => submitPrompt(prompt));
    suggestions.append(button);
  }

  section.append(title, subtitle, suggestions);
  return section;
}

function messageNode(message) {
  const article = document.createElement("article");
  article.className = `message ${message.role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = message.role === "user" ? "You" : "Aq";

  const body = document.createElement("div");
  body.className = "message-body";

  const head = document.createElement("div");
  head.className = "message-head";

  const role = document.createElement("span");
  role.className = "message-role";
  role.textContent = message.role === "user" ? "You" : "Aquaregia";
  head.append(role);

  if (message.role === "assistant") {
    const copy = document.createElement("button");
    copy.type = "button";
    copy.className = "copy-button";
    copy.textContent = "Copy";
    copy.disabled = !message.content;
    copy.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(message.content);
        copy.textContent = "Copied";
      } catch {
        copy.textContent = "Failed";
      }
      setTimeout(() => {
        copy.textContent = "Copy";
      }, 1200);
    });
    head.append(copy);
  }

  const content = document.createElement("div");
  content.className = "message-content";
  if (message.state === "streaming" && !message.content) {
    content.append(streamingDots());
  } else {
    content.textContent = message.content;
  }
  if (message.state === "error") content.classList.add("error");
  if (message.state === "stopped") content.classList.add("muted");

  body.append(head, content);
  article.append(avatar, body);
  return article;
}

function streamingDots() {
  const dots = document.createElement("span");
  dots.className = "typing";
  dots.innerHTML = "<span></span><span></span><span></span>";
  return dots;
}

function updateStreamingMessage(content) {
  const last = messagesEl.querySelector(".message.assistant:last-child .message-content");
  if (last) {
    last.textContent = content;
    scrollToBottom();
  } else {
    renderMessages();
  }
}

function setBusy(isBusy) {
  sendEl.disabled = isBusy;
  stopEl.hidden = !isBusy;
  promptEl.disabled = isBusy;
  turnStatusEl.textContent = isBusy ? "Generating" : "Ready";
}

function autosizePrompt() {
  promptEl.style.height = "auto";
  promptEl.style.height = `${Math.min(promptEl.scrollHeight, 180)}px`;
}

function activeThread() {
  return threads.find((thread) => thread.id === activeThreadId) ?? threads[0];
}

function createThread() {
  const thread = newThread();
  threads.unshift(thread);
  saveThreads();
  return thread;
}

function newThread() {
  return {
    id: crypto.randomUUID(),
    title: "New chat",
    messages: [],
    updatedAt: Date.now(),
  };
}

function loadThreads() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) && parsed.length > 0 ? parsed : [];
  } catch {
    return [];
  }
}

function saveThreads() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(threads.slice(0, 20)));
}

function titleFrom(content) {
  return content.length > 42 ? `${content.slice(0, 42)}...` : content;
}

function threadSubtitle(thread) {
  const count = thread.messages.filter((message) => message.role === "user").length;
  if (count === 0) return "No messages";
  return `${count} ${count === 1 ? "message" : "messages"}`;
}

function usageLabel(data) {
  try {
    const usage = JSON.parse(data);
    return `Tokens ${usage.total}`;
  } catch {
    return "Generating";
  }
}

function closeSidebarOnSmallScreen() {
  document.body.classList.remove("sidebar-open");
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function* readSse(body) {
  const reader = body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;

    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const event = parseSseFrame(frame);
      if (event) yield event;
    }
  }
}

function parseSseFrame(frame) {
  let name = "message";
  const data = [];

  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) {
      name = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      data.push(sseValue(line.slice(5)));
    }
  }

  return data.length ? { name, data: data.join("\n") } : null;
}

function sseValue(value) {
  const withoutReturn = value.endsWith("\r") ? value.slice(0, -1) : value;
  return withoutReturn.startsWith(" ") ? withoutReturn.slice(1) : withoutReturn;
}
