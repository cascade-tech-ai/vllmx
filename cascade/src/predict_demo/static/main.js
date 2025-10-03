const configInfo = document.getElementById("config-info");
const layout = document.querySelector(".layout");
const documentInput = document.getElementById("document-input");
const outputView = document.getElementById("output-view");
const tabButtons = document.querySelectorAll(".tab-button");
const panels = document.querySelectorAll(".panel");
const promptInput = document.getElementById("prompt-input");
const submitButton = document.getElementById("submit-button");
const statusText = document.getElementById("status-text");
const sampleSelect = document.getElementById("sample-select");
const loadSampleButton = document.getElementById("load-sample");
const tempSlider = document.getElementById("temperature");
const tempValueLabel = document.getElementById("temperature-value");
const usePredictionCheckbox = document.getElementById("use-prediction");
const promptBar = document.querySelector(".prompt-bar");
const tryItNowButton = document.getElementById("try-it-now");
const clearButton = document.getElementById("clear-fields");

const statFields = {
  chunks: document.getElementById("stat-chunks"),
  tokens: document.getElementById("stat-tokens"),
  accepted: document.getElementById("stat-accepted"),
  proposed: document.getElementById("stat-proposed"),
  acceptanceTotal: document.getElementById("stat-acceptance-total"),
  acceptanceProposed: document.getElementById("stat-acceptance-proposed"),
  promptTime: document.getElementById("stat-prompt-time"),
  generationTime: document.getElementById("stat-generation-time"),
  totalTime: document.getElementById("stat-total-time"),
};

const state = {
  activeTab: "welcome",
  streaming: false,
  model: null,
  serverUrl: null,
  defaultSample: null,
  samplePrompts: {},
  stats: resetStats(),
};

function shouldAutoFocus() {
  // if (!window.matchMedia) {
  //   return true;
  // }
  // const prefersCoarsePointer = window.matchMedia("(pointer: coarse)").matches;
  // const lacksHover = window.matchMedia("(hover: none)").matches;
  // return !(prefersCoarsePointer && lacksHover);
  return false;
}

function resetStats() {
  return {
    chunks: 0,
    tokens: 0,
    accepted: 0,
    proposed: 0,
    promptTimeMs: 0,
    generationTimeMs: 0,
    totalTimeMs: 0,
    startTime: null,
    firstTokenTime: null,
  };
}

function setActiveTab(tab) {
  state.activeTab = tab;
  tabButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tab);
  });
  panels.forEach((panel) => {
    const isActive = panel.dataset.panel === tab;
    panel.classList.toggle("active", isActive);
    panel.hidden = !isActive;
  });
  const isWelcome = tab === "welcome";
  if (promptBar) {
    promptBar.classList.toggle("hidden", isWelcome);
  }
  if (layout) {
    layout.classList.toggle("no-sidebar", isWelcome);
  }
  if (isWelcome) {
    updateStatus("");
  }
  if (!isWelcome && !state.streaming && shouldAutoFocus()) {
    promptInput.focus();
  }
  refreshSubmitState();
}

function refreshSubmitState() {
  const hasPrompt = Boolean(promptInput.value.trim());
  const enabled = !state.streaming && hasPrompt;
  submitButton.disabled = !enabled;
}

function updateStatus(message, isError = false) {
  statusText.textContent = message;
  statusText.classList.toggle("error", isError);
}

function escapeHtml(value) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function updateStatsDisplay() {
  statFields.chunks.textContent = state.stats.chunks.toString();
  statFields.tokens.textContent = state.stats.tokens.toString();
  statFields.accepted.textContent = state.stats.accepted.toString();
  statFields.proposed.textContent = state.stats.proposed.toString();

  const tokenPct = state.stats.tokens
    ? (state.stats.accepted / state.stats.tokens) * 100
    : 0;
  const proposedPct = state.stats.proposed
    ? (state.stats.accepted / state.stats.proposed) * 100
    : 0;

  statFields.acceptanceTotal.textContent = `${tokenPct.toFixed(1)}%`;
  statFields.acceptanceProposed.textContent = `${proposedPct.toFixed(1)}%`;

  const promptSeconds = state.stats.promptTimeMs / 1000;
  const generationSeconds = state.stats.generationTimeMs / 1000;
  const totalSeconds = state.stats.totalTimeMs / 1000;

  statFields.promptTime.textContent = `${promptSeconds.toFixed(2)}s`;
  statFields.generationTime.textContent = `${generationSeconds.toFixed(2)}s`;
  statFields.totalTime.textContent = `${totalSeconds.toFixed(2)}s`;
}

async function fetchConfig() {
  try {
    const resp = await fetch("/api/config");
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }
    const data = await resp.json();
    state.model = data.model;
    state.serverUrl = data.server_url;
    state.defaultSample = data.default_sample || null;
    state.samplePrompts = data.sample_prompts || {};
    if (configInfo) {
      if (data.model) {
        configInfo.textContent = "";
        configInfo.setAttribute("data-model", data.model);
        configInfo.setAttribute("title", data.model);
      } else {
        configInfo.textContent = "";
        configInfo.removeAttribute("data-model");
        configInfo.removeAttribute("title");
      }
    }
  } catch (error) {
    console.error("Failed to load config", error);
    updateStatus("Failed to load config", true);
  }
}

async function fetchSamples() {
  try {
    const resp = await fetch("/api/samples");
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }
    const data = await resp.json();
    const samples = data.samples || [];
    sampleSelect.innerHTML = '<option value="">Select sample…</option>';
    for (const sample of samples) {
      const option = document.createElement("option");
      option.value = sample.id;
      option.textContent = sample.label || sample.id;
      sampleSelect.appendChild(option);
    }
    if (!state.defaultSample && typeof data.default === "string") {
      state.defaultSample = data.default;
    }
    if (data.sample_prompts && typeof data.sample_prompts === "object") {
      state.samplePrompts = { ...state.samplePrompts, ...data.sample_prompts };
    }
    return samples;
  } catch (error) {
    console.error("Failed to load samples", error);
    updateStatus("Failed to load samples", true);
    return [];
  }
}

async function loadSample(sampleId = null, { auto = false } = {}) {
  const targetId = sampleId ?? sampleSelect.value;
  if (!targetId) {
    if (!auto) {
      updateStatus("Select a sample first");
    }
    return;
  }
  if (!auto) {
    loadSampleButton.disabled = true;
  }
  try {
    const resp = await fetch(`/api/samples/${encodeURIComponent(targetId)}`);
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }
    const data = await resp.json();
    documentInput.value = data.content || "";
    const prompt = state.samplePrompts?.[targetId] || "";
    promptInput.value = prompt;
    resetOutput();
    if (!auto) {
      setActiveTab("input");
    }
    updateStatus(`Loaded ${data.label || targetId}`);
    if (sampleSelect.value !== targetId) {
      sampleSelect.value = targetId;
    }
  } catch (error) {
    console.error("Failed to load sample", error);
    updateStatus("Failed to load sample", true);
  } finally {
    if (!auto) {
      loadSampleButton.disabled = false;
    }
    refreshSubmitState();
  }
}

function resetOutput() {
  outputView.innerHTML = "";
  state.stats = resetStats();
  updateStatsDisplay();
}

function parseSSEEvent(raw) {
  if (!raw) {
    return null;
  }
  const lines = raw.split("\n");
  const dataLines = [];
  for (const line of lines) {
    if (line.startsWith(":")) {
      continue; // comment line
    }
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }
  if (!dataLines.length) {
    return null;
  }
  return dataLines.join("\n");
}

async function processChunk(chunk) {
  const choices = chunk?.choices;
  if (!Array.isArray(choices) || !choices.length) {
    return;
  }
  const choice = choices[0];
  const delta = choice.delta || {};
  const segments = [];
  const deltaContent = delta.content;
  if (typeof deltaContent === "string") {
    segments.push(deltaContent);
  } else if (Array.isArray(deltaContent)) {
    for (const part of deltaContent) {
      if (part && typeof part === "object" && "text" in part) {
        segments.push(String(part.text));
      }
    }
  }
  const chunkText = segments.join("");
  if (!chunkText) {
    return;
  }
  if (state.stats.startTime == null) {
    state.stats.startTime = performance.now();
  }
  if (state.stats.firstTokenTime == null) {
    state.stats.firstTokenTime = performance.now();
  }

  const proposedValue = Number(choice.spec_tokens_proposed);
  const acceptedTokensValue = Number(choice.spec_tokens_accepted);
  const proposedCount = Number.isFinite(proposedValue) && proposedValue > 0
    ? proposedValue
    : 0;
  const acceptedTokens = Number.isFinite(acceptedTokensValue) && acceptedTokensValue > 0
    ? acceptedTokensValue
    : 0;
  const acceptedCharsValue = Number(choice.spec_chars_accepted);
  const acceptedChars = Number.isFinite(acceptedCharsValue) && acceptedCharsValue > 0
    ? acceptedCharsValue
    : 0;
  const displayText = chunkText;
  const safeDisplay = escapeHtml(displayText);
  const displayLength = displayText.length;
  const boundedAcceptedChars = Math.max(0, Math.min(acceptedChars, displayLength));

  let html = "";
  if (!displayText) {
    html = "";
  } else if (!proposedCount) {
    html = `<span class="token-neutral">${safeDisplay}</span>`;
  } else if (boundedAcceptedChars === 0) {
    html = acceptedTokens > 0
      ? `<span class="token-neutral">${safeDisplay}</span>`
      : `<span class="token-rejected">${safeDisplay}</span>`;
  } else if (boundedAcceptedChars >= displayLength) {
    html = `<span class="token-accepted">${safeDisplay}</span>`;
  } else {
    const acceptedSlice = displayText.slice(0, boundedAcceptedChars);
    const remainder = displayText.slice(boundedAcceptedChars);
    html =
      `<span class="token-accepted">${escapeHtml(acceptedSlice)}</span>` +
      `<span class="token-rejected">${escapeHtml(remainder)}</span>`;
  }

  outputView.insertAdjacentHTML("beforeend", html);
  outputView.scrollTop = outputView.scrollHeight;

  state.stats.chunks += 1;
  const countedTokens = proposedCount || Math.max(acceptedTokens, displayLength);
  state.stats.tokens += countedTokens;
  state.stats.accepted += Math.max(0, Math.min(acceptedTokens, countedTokens));
  state.stats.proposed += proposedCount;
}

async function streamCompletion() {
  const payload = {
    document: documentInput.value,
    prompt: promptInput.value,
    temperature: Number(tempSlider.value),
    use_prediction: usePredictionCheckbox.checked,
  };

  const response = await fetch("/api/chat-stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok || !response.body) {
    const text = await response.text();
    throw new Error(text || `Request failed with ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let doneStreaming = false;
  state.stats.startTime = performance.now();

  while (!doneStreaming) {
    const { value, done } = await reader.read();
    if (done) {
      doneStreaming = true;
      buffer += decoder.decode();
    } else {
      buffer += decoder.decode(value, { stream: true });
    }

    let eventEnd = buffer.indexOf("\n\n");
    while (eventEnd !== -1) {
      const rawEvent = buffer.slice(0, eventEnd);
      buffer = buffer.slice(eventEnd + 2);
      const dataPayload = parseSSEEvent(rawEvent);
      if (dataPayload) {
        if (dataPayload === "[DONE]") {
          doneStreaming = true;
          break;
        }
        try {
          const parsed = JSON.parse(dataPayload);
          await processChunk(parsed);
        } catch (error) {
          console.warn("Failed to parse chunk", error);
        }
      }
      eventEnd = buffer.indexOf("\n\n");
    }
  }

  if (state.stats.firstTokenTime != null) {
    state.stats.promptTimeMs =
      state.stats.firstTokenTime - state.stats.startTime;
    state.stats.generationTimeMs =
      performance.now() - state.stats.firstTokenTime;
  }
  state.stats.totalTimeMs = performance.now() - state.stats.startTime;
}

async function submitPrompt() {
  if (state.streaming) {
    return;
  }
  if (!promptInput.value.trim()) {
    updateStatus("Enter a prompt", true);
    return;
  }
  state.streaming = true;
  resetOutput();
  setActiveTab("output");
  updateStatus("Streaming…");
  refreshSubmitState();

  try {
    await streamCompletion();
    updateStatus("Completed");
  } catch (error) {
    console.error("Streaming failed", error);
    updateStatus("Streaming failed", true);
  } finally {
    state.streaming = false;
    if (state.stats.firstTokenTime == null) {
      state.stats.promptTimeMs = 0;
      state.stats.generationTimeMs = 0;
    }
    updateStatsDisplay();
    refreshSubmitState();
  }
}

function registerEvents() {
  tabButtons.forEach((button) =>
    button.addEventListener("click", () => {
      if (state.streaming) {
        return;
      }
      setActiveTab(button.dataset.tab);
      updateStatus("");
    })
  );

  if (tryItNowButton) {
    tryItNowButton.addEventListener("click", () => {
      setActiveTab("input");
    });
  }

  promptInput.addEventListener("input", refreshSubmitState);
  promptInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      submitPrompt();
    }
  });

  documentInput.addEventListener("input", () => {
    if (!state.streaming) {
      updateStatus("");
    }
  });

  submitButton.addEventListener("click", submitPrompt);
  loadSampleButton.addEventListener("click", () => {
    void loadSample();
  });
  tempSlider.addEventListener("input", () => {
    tempValueLabel.textContent = Number(tempSlider.value).toFixed(2);
  });
  tempValueLabel.textContent = Number(tempSlider.value).toFixed(2);

  usePredictionCheckbox.addEventListener("change", () => {
    updateStatus(usePredictionCheckbox.checked ? "Prediction on" : "Prediction off");
  });

  if (clearButton) {
    clearButton.addEventListener("click", () => {
      if (state.streaming) {
        return;
      }
      documentInput.value = "";
      promptInput.value = "";
      resetOutput();
      updateStatus("");
      refreshSubmitState();
      if (state.activeTab !== "welcome" && shouldAutoFocus()) {
        promptInput.focus();
      }
    });
  }
}

async function init() {
  registerEvents();
  setActiveTab(state.activeTab);
  updateStatsDisplay();
  await fetchConfig();
  const samples = await fetchSamples();
  const defaultSample = state.defaultSample;
  if (defaultSample && samples.some((item) => item.id === defaultSample)) {
    sampleSelect.value = defaultSample;
    await loadSample(defaultSample, { auto: true });
  } else if (samples.length) {
    const fallback = samples[0].id;
    sampleSelect.value = fallback;
    await loadSample(fallback, { auto: true });
  }
  updateStatus("Ready");
  refreshSubmitState();
}

init().catch((error) => {
  console.error("Initialization failed", error);
  updateStatus("Initialization failed", true);
});
