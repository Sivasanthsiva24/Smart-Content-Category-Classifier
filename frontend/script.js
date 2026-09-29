const input = document.querySelector("#content-input");
const classifyButton = document.querySelector("#classify-button");
const clearButton = document.querySelector("#clear-button");
const formMessage = document.querySelector("#form-message");
const characterCount = document.querySelector("#character-count");
const resultContent = document.querySelector("#result-content");
const resultState = document.querySelector("#result-state");
const historyBody = document.querySelector("#history-body");
const clearHistoryButton = document.querySelector("#clear-history-button");
const historyKey = "signal-classifier-history";
const historyLimit = 8;

function getHistory() {
  try {
    const items = JSON.parse(localStorage.getItem(historyKey) || "[]");
    return Array.isArray(items) ? items : [];
  } catch {
    return [];
  }
}

function setLoading(loading) {
  classifyButton.disabled = loading;
  input.disabled = loading;
  classifyButton.classList.toggle("is-loading", loading);
  classifyButton.querySelector(".button-label").textContent = loading ? "Classifying" : "Classify content";
  resultState.textContent = loading ? "ANALYZING TEXT" : "AWAITING TEXT";
}

function makeElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function showError(message) {
  formMessage.textContent = message;
  input.setAttribute("aria-invalid", "true");
  input.focus();
}

function renderResult(result) {
  resultState.textContent = "PREDICTION READY";
  resultContent.replaceChildren();

  const grid = makeElement("div", "prediction-grid");
  const category = makeElement("div", "prediction-category");
  category.append(makeElement("div", "result-kicker", "Predicted category"));
  category.append(makeElement("div", "category-name", result.category));

  const confidence = makeElement("div", "confidence-block");
  const confidenceValue = makeElement("div", "confidence-value");
  confidenceValue.append(makeElement("span", "", `${result.confidence}%`));
  confidenceValue.append(makeElement("small", "", "MODEL CONFIDENCE"));
  const track = makeElement("div", "confidence-track");
  track.setAttribute("role", "meter");
  track.setAttribute("aria-label", "Prediction confidence");
  track.setAttribute("aria-valuemin", "0");
  track.setAttribute("aria-valuemax", "100");
  track.setAttribute("aria-valuenow", String(result.confidence));
  const fill = makeElement("div", "confidence-fill");
  fill.style.width = `${Math.max(0, Math.min(100, result.confidence))}%`;
  track.append(fill);
  confidence.append(makeElement("div", "detail-label", "Confidence"), confidenceValue, track);

  const explanation = makeElement("div", "explanation-block");
  explanation.append(makeElement("div", "detail-label", "Why this category"));
  explanation.append(makeElement("p", "", result.explanation));
  explanation.append(makeElement("div", "detail-label", "Extracted keywords"));
  const keywords = makeElement("div", "keyword-list");
  for (const keyword of result.keywords || []) keywords.append(makeElement("span", "keyword", keyword));
  if (!result.keywords?.length) keywords.append(makeElement("span", "keyword", "No distinctive keywords"));
  explanation.append(keywords);

  grid.append(category, confidence, explanation);
  resultContent.append(grid);
}

function renderHistory() {
  const items = getHistory();
  historyBody.replaceChildren();
  clearHistoryButton.hidden = items.length === 0;
  if (!items.length) {
    const row = makeElement("tr", "history-empty");
    const cell = makeElement("td", "", "Your recent predictions will appear here.");
    cell.colSpan = 4;
    row.append(cell);
    historyBody.append(row);
    return;
  }

  for (const item of items) {
    const row = document.createElement("tr");
    const textCell = document.createElement("td");
    const text = makeElement("span", "history-text", item.text);
    text.title = item.text;
    textCell.append(text);
    row.append(textCell);
    row.append(makeElement("td", "history-category", item.category));
    row.append(makeElement("td", "history-confidence", `${item.confidence}%`));
    row.append(makeElement("td", "history-time", new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" }).format(new Date(item.timestamp))));
    historyBody.append(row);
  }
}

function saveToHistory(text, result) {
  const items = getHistory();
  items.unshift({ text, category: result.category, confidence: result.confidence, timestamp: new Date().toISOString() });
  try {
    localStorage.setItem(historyKey, JSON.stringify(items.slice(0, historyLimit)));
  } catch {
    formMessage.textContent = "Prediction complete, but browser storage is unavailable for history.";
  }
  renderHistory();
}

async function classifyContent() {
  const text = input.value.trim();
  formMessage.textContent = "";
  input.removeAttribute("aria-invalid");
  if (!text) {
    showError("Please enter some content before classifying.");
    return;
  }

  setLoading(true);
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "The content could not be classified.");
    renderResult(result);
    saveToHistory(text, result);
  } catch (error) {
    formMessage.textContent = error instanceof TypeError
      ? "Could not reach the classifier. Check that the Flask server is running and try again."
      : error.message;
    resultState.textContent = "CLASSIFICATION FAILED";
  } finally {
    setLoading(false);
  }
}

input.addEventListener("input", () => {
  characterCount.textContent = `${input.value.length.toLocaleString()} / 12,000`;
  formMessage.textContent = "";
  input.removeAttribute("aria-invalid");
});
classifyButton.addEventListener("click", classifyContent);
input.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") classifyContent();
});
clearButton.addEventListener("click", () => {
  input.value = "";
  characterCount.textContent = "0 / 12,000";
  formMessage.textContent = "";
  input.removeAttribute("aria-invalid");
  resultState.textContent = "AWAITING TEXT";
  resultContent.innerHTML = '<div class="empty-result"><span class="empty-mark" aria-hidden="true">?</span><div><h3>Your result will land here.</h3><p>Add a sentence or article above to see its predicted topic, confidence, and keywords.</p></div></div>';
  input.focus();
});
for (const button of document.querySelectorAll(".example-button")) {
  button.addEventListener("click", () => {
    input.value = button.dataset.example;
    characterCount.textContent = `${input.value.length.toLocaleString()} / 12,000`;
    formMessage.textContent = "";
    classifyContent();
  });
}
clearHistoryButton.addEventListener("click", () => {
  localStorage.removeItem(historyKey);
  renderHistory();
});
renderHistory();