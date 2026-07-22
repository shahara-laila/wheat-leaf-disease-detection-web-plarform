/* Wheat Leaf Disease Detection — frontend. No framework, no build step. */
"use strict";

var MAX_BYTES = 10 * 1024 * 1024;
var $ = function (id) { return document.getElementById(id); };

function esc(s) {
  return String(s).replace(/[&<>"']/g, function (c) {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
  });
}
function pretty(n) { return String(n).replace(/_/g, " "); }

/** Read an error body that may be JSON or plain text, and never throw. */
function errorText(res) {
  return res.text().then(function (body) {
    try {
      var j = JSON.parse(body);
      return j.detail ? j.error + " — " + j.detail : (j.error || body);
    } catch (e) { return body || res.status + " " + res.statusText; }
  });
}

/* ── tabs ─────────────────────────────────────────────── */
var tabs = [$("tab-image"), $("tab-symptom"), $("tab-browse")];

function selectTab(i) {
  tabs.forEach(function (t, j) {
    var on = i === j;
    t.setAttribute("aria-selected", String(on));
    t.tabIndex = on ? 0 : -1;
    $(t.getAttribute("aria-controls")).hidden = !on;
  });
  if (i === 2) loadDiseases();
}

tabs.forEach(function (tab, i) {
  tab.addEventListener("click", function () { selectTab(i); });
  tab.addEventListener("keydown", function (e) {
    var d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
    if (!d) return;
    e.preventDefault();
    var n = (i + d + tabs.length) % tabs.length;
    selectTab(n);
    tabs[n].focus();
  });
});

/* ── health ───────────────────────────────────────────── */
var modelReady = false;

fetch("/health")
  .then(function (r) { return r.json(); })
  .then(function (h) {
    modelReady = h.model_loaded;
    var line = $("status-line");
    if (modelReady) {
      line.textContent = h.kb_entries + " diseases · model ready";
    } else {
      line.textContent = "Image detection unavailable — " + (h.model_error || "model not loaded");
      line.className = "status bad";
      var z = $("drop-zone");
      z.setAttribute("aria-disabled", "true");
      z.tabIndex = -1;
      $("predict-result").innerHTML =
        '<p class="warn">The model is not loaded. Symptom search and the disease ' +
        "list still work.</p>";
    }
  })
  .catch(function () {
    var line = $("status-line");
    line.textContent = "Cannot reach the API — is the server running?";
    line.className = "status bad";
  });

/* ── image ────────────────────────────────────────────── */
var dropZone = $("drop-zone");
var fileInput = $("file-input");
var previewWrap = $("preview-wrap");
var previewImg = $("preview");
var imageError = $("image-error");
var objectUrl = null;

function showError(msg) { imageError.textContent = msg; imageError.hidden = false; }
function clearError() { imageError.hidden = true; }

function resetPreview() {
  // Revoking is the visible counterpart of the no-persistence policy.
  if (objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl = null; }
  previewWrap.hidden = true;
  previewImg.removeAttribute("src");
}

$("clear-btn").addEventListener("click", function () {
  resetPreview();
  fileInput.value = "";
  $("predict-result").innerHTML = "";
  clearError();
});

function handleFile(file) {
  clearError();
  if (!file) return;
  if (!modelReady) { showError("The model is not loaded on the server."); return; }
  // Fail fast rather than spending a round trip.
  if (file.type && file.type.indexOf("image/") !== 0) {
    showError("That is not an image file."); return;
  }
  if (file.size > MAX_BYTES) {
    showError("Image is " + (file.size / 1048576).toFixed(1) + " MB; the limit is 10 MB."); return;
  }
  resetPreview();
  objectUrl = URL.createObjectURL(file);
  previewImg.src = objectUrl;
  previewWrap.hidden = false;
  classify(file);
}

function classify(file) {
  var out = $("predict-result");
  out.innerHTML = '<p class="sub">Classifying…</p>';
  var form = new FormData();
  form.append("image", file);

  fetch("/predict", { method: "POST", body: form })
    .then(function (res) {
      if (!res.ok) return errorText(res).then(function (m) { throw new Error(m); });
      return res.json();
    })
    .then(render)
    .catch(function (err) {
      out.innerHTML = '<p class="error">' + esc(err.message) + "</p>";
    });

  function render(d) {
    var names = Object.keys(d.probabilities).sort(function (a, b) {
      return d.probabilities[b] - d.probabilities[a];
    });
    var bars = names.map(function (n) {
      var pct = d.probabilities[n] * 100;
      return (
        '<div class="bar-row' + (n === d.prediction ? " top" : "") +
          '" role="img" aria-label="' + esc(pretty(n)) + " " + pct.toFixed(1) + ' percent">' +
          '<span class="bar-label">' + esc(pretty(n)) + "</span>" +
          '<span class="bar-track"><span class="bar-fill" style="width:' + pct.toFixed(1) + '%"></span></span>' +
          '<span class="bar-value">' + pct.toFixed(0) + "%</span>" +
        "</div>"
      );
    }).join("");

    out.innerHTML =
      '<p class="headline">' + esc(pretty(d.prediction)) + "</p>" +
      '<p class="sub">' + (d.confidence * 100).toFixed(1) + "% confidence" +
        (d.uncertain ? " · low" : "") + "</p>" +
      bars +
      (d.uncertain ? '<p class="warn">' + esc(d.message || "") + "</p>" : "") +
      '<p class="treatment">' + esc(d.recommendation) + "</p>";
  }
}

dropZone.addEventListener("click", function () { if (modelReady) fileInput.click(); });
dropZone.addEventListener("keydown", function (e) {
  if ((e.key === "Enter" || e.key === " ") && modelReady) { e.preventDefault(); fileInput.click(); }
});
fileInput.addEventListener("change", function () { handleFile(fileInput.files[0]); });

["dragenter", "dragover"].forEach(function (ev) {
  dropZone.addEventListener(ev, function (e) { e.preventDefault(); dropZone.classList.add("dragging"); });
});
["dragleave", "drop"].forEach(function (ev) {
  dropZone.addEventListener(ev, function (e) { e.preventDefault(); dropZone.classList.remove("dragging"); });
});
dropZone.addEventListener("drop", function (e) {
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

/* ── symptom ──────────────────────────────────────────── */
var symptomInput = $("symptom-input");

Array.prototype.forEach.call(document.querySelectorAll(".example"), function (b) {
  b.addEventListener("click", function () { symptomInput.value = b.textContent; search(); });
});
$("symptom-btn").addEventListener("click", search);
symptomInput.addEventListener("keydown", function (e) {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) search();
});

function search() {
  var text = symptomInput.value.trim();
  var out = $("symptom-result");
  if (!text) { out.innerHTML = '<p class="sub">Type a symptom first.</p>'; return; }
  out.innerHTML = '<p class="sub">Searching…</p>';

  fetch("/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: text })
  })
    .then(function (res) {
      if (!res.ok) return errorText(res).then(function (m) { throw new Error(m); });
      return res.json();
    })
    .then(function (d) {
      if (!d.diseases.length) { out.innerHTML = '<p class="count">' + esc(d.message) + "</p>"; return; }
      // Shared symptoms mean candidates, not a diagnosis. Say so.
      var warn = d.ambiguous
        ? '<p class="warn">' + d.diseases.length +
          " diseases share this symptom — add detail to narrow it down.</p>"
        : "";
      out.innerHTML = warn + d.recommendations.map(function (r) {
        return (
          '<div class="entry">' +
            "<h3>" + esc(pretty(r.disease)) + "</h3>" +
            '<p class="meta">matched: ' + esc(r.matched_terms.join(", ")) + "</p>" +
            "<p>" + esc(r.recommendation) + "</p>" +
          "</div>"
        );
      }).join("");
    })
    .catch(function (err) {
      out.innerHTML = '<p class="error">' + esc(err.message) + "</p>";
    });
}

/* ── diseases ─────────────────────────────────────────── */
var cache = null;

function loadDiseases() {
  if (cache) return;
  var out = $("browse-result");
  out.innerHTML = '<p class="sub">Loading…</p>';

  fetch("/diseases")
    .then(function (r) { return r.json(); })
    .then(function (data) {
      cache = Object.keys(data.diseases).map(function (name) {
        var d = data.diseases[name];
        return {
          name: name,
          photo: d.detectable_by_image,
          common: d.common_names,
          symptoms: d.symptoms,
          rec: d.recommendation,
          hay: (name + " " + d.common_names.join(" ") + " " + d.symptoms.join(" ")).toLowerCase()
        };
      });
      renderList("");
    })
    .catch(function () { out.innerHTML = '<p class="error">Could not load diseases.</p>'; });
}

function renderList(q) {
  var query = q.trim().toLowerCase();
  var hits = cache.filter(function (d) { return !query || d.hay.indexOf(query) !== -1; });

  $("browse-result").innerHTML =
    '<p class="count">' + hits.length + " of " + cache.length + " diseases</p>" +
    (hits.length
      ? hits.map(function (d) {
          return (
            '<div class="entry">' +
              "<h3>" + esc(pretty(d.name)) +
                (d.photo ? '<span class="tag">photo</span>' : "") + "</h3>" +
              '<p class="meta">' + esc(d.symptoms.join(", ")) + "</p>" +
              "<p>" + esc(d.rec) + "</p>" +
            "</div>"
          );
        }).join("")
      : "");
}

$("browse-filter").addEventListener("input", function (e) {
  if (cache) renderList(e.target.value);
});
