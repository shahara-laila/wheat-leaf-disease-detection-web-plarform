

const $ = (id) => document.getElementById(id);
const pretty = (name) => name.replace(/_/g, " ");
const percent = (value) => (value * 100).toFixed(1) + "%";



async function showStatus() {
  try {
    const response = await fetch("/health");
    const health = await response.json();
    $("status").textContent = health.model_loaded
      ? `Ready. ${health.disease_count} diseases in the database.`
      : "Photo checking is off (model not loaded). Symptom search still works.";
  } catch (error) {
    $("status").textContent = "Cannot reach the server.";
  }
}


// Both the file button and the drop zone end up calling this one function.
async function classifyPhoto(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    $("predict-result").textContent = "Please choose an image file.";
    return;
  }

  $("preview").src = URL.createObjectURL(file);
  $("preview").hidden = false;
  $("predict-result").innerHTML = '<p class="loading"><span class="spinner"></span>Checking...</p>';

  const form = new FormData();
  form.append("image", file);

  try {
    const response = await fetch("/predict", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) {
      // FastAPI puts the error message in "detail".
      $("predict-result").textContent = data.detail;
      return;
    }
    showPrediction(data);
  } catch (error) {
    $("predict-result").textContent = "Cannot reach the server.";
  }
}


$("file-input").addEventListener("change", (event) => {
  classifyPhoto(event.target.files[0]);
});


const dropZone = $("drop-zone");
dropZone.addEventListener("click", () => $("file-input").click());
dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();                 // required, or the browser just opens the file
  dropZone.classList.add("dragover");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragover");
  classifyPhoto(event.dataTransfer.files[0]);
});

function showPrediction(data) {
  const bars = Object.entries(data.probabilities)
    .sort((a, b) => b[1] - a[1])
    .map(([name, value]) => `
      <div class="bar">
        <span>${pretty(name)}</span>
        <progress value="${value}" max="1"></progress>
        <span>${percent(value)}</span>
      </div>`)
    .join("");


  const healthy = data.prediction === "Healthy" && !data.uncertain;
  const colour = healthy ? "good" : "alert";

  const warning = data.uncertain
    ? `<p class="warning">
         Low confidence. The model only knows three diseases and still answers
         confidently on photos unlike its training images, so treat this result
         with care. Try a close-up of a single leaf.
       </p>`
    : "";

  $("predict-result").innerHTML = `
    <div class="result ${colour}">
      <h3>${pretty(data.prediction)}</h3>
      <p class="confidence">${percent(data.confidence)} confident</p>
    </div>
    ${bars}
    ${warning}
    <p>${data.recommendation}</p>`;
}


$("search-button").addEventListener("click", search);
$("symptom-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter") search();
});

async function search() {
  const text = $("symptom-input").value.trim();
  if (!text) return;

  $("symptom-result").textContent = "Searching...";

  try {
    const response = await fetch("/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text }),
    });
    const data = await response.json();

    if (data.matches.length === 0) {
      $("symptom-result").textContent =
        "No disease matched. Try describing the symptom differently.";
      return;
    }

    $("symptom-result").innerHTML = data.matches
      .map((match) => `
        <div class="entry">
          <h3>${pretty(match.disease)}</h3>
          <p class="matched">matched: ${match.matched_terms.join(", ")}</p>
          <p>${match.recommendation}</p>
        </div>`)
      .join("");
  } catch (error) {
    $("symptom-result").textContent = "Cannot reach the server.";
  }
}


async function showDiseases() {
  try {
    const response = await fetch("/diseases");
    const diseases = await response.json();

    $("disease-list").innerHTML = Object.entries(diseases)
      .map(([name, info]) => `
        <div class="entry">
          <h3>${pretty(name)}</h3>
          <p class="matched">symptoms: ${info.symptoms.join(", ")}</p>
          <p>${info.recommendation}</p>
        </div>`)
      .join("");
  } catch (error) {
    $("disease-list").textContent = "Could not load the disease list.";
  }
}

showStatus();
showDiseases();
