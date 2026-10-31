const $ = (id) => document.getElementById(id);
const fileInput = $("file"), drop = $("drop"), preview = $("preview"), hint = $("drop-hint");
const queryInput = $("query"), goBtn = $("go"), statusEl = $("status"), resultsEl = $("results"), sugEl = $("suggestions");

let currentFile = null;

function setFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  currentFile = file;
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  hint.hidden = true;
  goBtn.disabled = false;
  queryInput.value = "";
  sugEl.innerHTML = "";
  resultsEl.innerHTML = "";
  statusEl.textContent = "";
}

fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));
window.addEventListener("paste", (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (item) setFile(item.getAsFile());
});

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function renderResults(data) {
  resultsEl.innerHTML = "";
  for (const r of data.results) {
    const card = el("a", "card");
    card.href = r.url;
    card.target = "_blank";
    card.rel = "noopener noreferrer";
    const img = el("img");
    img.src = r.image_url;
    img.loading = "lazy";
    img.alt = r.title;
    const meta = el("div", "meta");
    meta.append(
      el("div", "price", r.price ? `${r.currency === "USD" ? "$" : r.currency + " "}${r.price}` : "—"),
      el("div", "title", r.title),
      el("div", "score", `${Math.round(r.score * 100)}% match · ${r.source}`),
    );
    card.append(img, meta);
    resultsEl.append(card);
  }
  sugEl.innerHTML = "";
  for (const s of data.suggested_queries) {
    const b = el("button", "", s);
    b.type = "button";
    b.addEventListener("click", () => { queryInput.value = s; search(); });
    sugEl.append(b);
  }
}

async function search() {
  if (!currentFile) return;
  goBtn.disabled = true;
  statusEl.className = "";
  statusEl.textContent = "Searching and comparing images…";
  resultsEl.innerHTML = "";
  const body = new FormData();
  body.append("image", currentFile);
  if (queryInput.value.trim()) body.append("query", queryInput.value.trim());
  try {
    const resp = await fetch("/api/search", { method: "POST", body });
    if (!resp.ok) throw new Error((await resp.json().catch(() => ({}))).detail || resp.statusText);
    const data = await resp.json();
    queryInput.value = data.query;
    renderResults(data);
    statusEl.textContent = `Ranked ${data.candidates_considered} listings for “${data.query}”.`;
  } catch (err) {
    statusEl.className = "error";
    statusEl.textContent = `Error: ${err.message}`;
  } finally {
    goBtn.disabled = false;
  }
}

$("form").addEventListener("submit", (e) => { e.preventDefault(); search(); });
