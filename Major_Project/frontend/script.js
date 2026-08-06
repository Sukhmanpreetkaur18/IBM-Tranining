// Smart API Endpoint Selection
const API_BASE = (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1" || window.location.protocol === "file:")
    ? "http://127.0.0.1:8000"
    : window.location.origin;

document.addEventListener("DOMContentLoaded", () => {
    // Tab Navigation
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
            btn.classList.add("active");
            document.getElementById(`panel-${btn.dataset.tab}`).classList.add("active");
        });
    });

    function showError(el, msg) { if (el) el.textContent = msg; }
    function clearError(el) { if (el) el.textContent = ""; }

    async function parseResponse(res) {
        if (!res.ok) {
            let errorDetail = `Request failed (${res.status} ${res.statusText})`;
            try {
                const errData = await res.json();
                if (errData && errData.detail) errorDetail = errData.detail;
            } catch (e) {
                // Ignore JSON parse errors for non-JSON 502/504 responses
            }
            throw new Error(errorDetail);
        }
        return await res.json();
    }

    // 1. Detect Tab Listener
    const detectBtn = document.getElementById("detectBtn");
    if (detectBtn) {
        detectBtn.addEventListener("click", async () => {
            const text = document.getElementById("detectInput").value.trim();
            const errorEl = document.getElementById("detectError");
            const resultEl = document.getElementById("detectResult");
            clearError(errorEl);
            resultEl.classList.remove("visible");

            if (!text) { showError(errorEl, "Please enter some text."); return; }

            detectBtn.disabled = true;
            detectBtn.textContent = "Analyzing...";
            try {
                const res = await fetch(`${API_BASE}/predict`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ text })
                });
                const data = await parseResponse(res);

                resultEl.classList.remove("real", "fake");
                resultEl.classList.add(data.prediction.toLowerCase(), "visible");
                document.getElementById("detectLabel").textContent =
                    data.prediction === "Real" ? "✅ Likely Real" : "⚠️ Likely Fake";
                document.getElementById("detectConf").textContent = data.message;
            } catch (err) {
                showError(errorEl, `Error: ${err.message}`);
            } finally {
                detectBtn.disabled = false;
                detectBtn.textContent = "Analyze News";
            }
        });
    }

    // 2. Explain Tab Listener
    const explainBtn = document.getElementById("explainBtn");
    if (explainBtn) {
        explainBtn.addEventListener("click", async () => {
            const text = document.getElementById("explainInput").value.trim();
            const errorEl = document.getElementById("explainError");
            const resultEl = document.getElementById("explainResult");
            const wordsEl = document.getElementById("explainWords");
            clearError(errorEl);
            resultEl.classList.remove("visible");
            wordsEl.innerHTML = "";

            if (!text) { showError(errorEl, "Please enter some text."); return; }

            explainBtn.disabled = true;
            explainBtn.textContent = "Analyzing...";
            try {
                const res = await fetch(`${API_BASE}/explain`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ text })
                });
                const data = await parseResponse(res);

                resultEl.classList.remove("real", "fake");
                resultEl.classList.add(data.prediction.toLowerCase(), "visible");
                document.getElementById("explainLabel").textContent =
                    data.prediction === "Real" ? "✅ Likely Real" : "⚠️ Likely Fake";
                document.getElementById("explainConf").textContent = data.message;

                data.top_words.forEach(w => {
                    const tag = document.createElement("span");
                    tag.className = `word-tag ${w.leans}`;
                    tag.textContent = w.leans === "influential" ? `${w.word}` : `${w.word} → ${w.leans}`;
                    wordsEl.appendChild(tag);
                });
            } catch (err) {
                showError(errorEl, `Error: ${err.message}`);
            } finally {
                explainBtn.disabled = false;
                explainBtn.textContent = "Analyze & Explain";
            }
        });
    }

    // 3. Generate Tab Listener
    const generateBtn = document.getElementById("generateBtn");
    if (generateBtn) {
        generateBtn.addEventListener("click", async () => {
            const prompt = document.getElementById("generateInput").value.trim();
            const errorEl = document.getElementById("generateError");
            const resultEl = document.getElementById("generateResult");
            clearError(errorEl);
            resultEl.classList.remove("visible");

            if (!prompt) { showError(errorEl, "Please enter a headline or starting sentence."); return; }

            generateBtn.disabled = true;
            generateBtn.textContent = "Generating...";
            try {
                const res = await fetch(`${API_BASE}/generate`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ prompt, max_new_tokens: 50 })
                });
                const data = await parseResponse(res);

                resultEl.textContent = data.generated_text;
                resultEl.classList.add("visible");
            } catch (err) {
                showError(errorEl, `Error: ${err.message}`);
            } finally {
                generateBtn.disabled = false;
                generateBtn.textContent = "Generate News Text";
            }
        });
    }
});