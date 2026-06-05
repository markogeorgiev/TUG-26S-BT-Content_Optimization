const experimentArticleInput = document.querySelector("[data-experiment-article-input]");
const experimentArticlePageId = document.querySelector("[data-experiment-article-page-id]");
const experimentArticleSuggestions = document.querySelector("[data-experiment-article-suggestions]");
const experimentPickerForm = document.querySelector("[data-experiment-picker-form]");
const experimentPickerSubmit = document.querySelector("[data-experiment-picker-submit]");
const experimentRunForm = document.querySelector("[data-experiment-run-form]");
const experimentRunSubmit = document.querySelector("[data-experiment-run-submit]");

if (experimentArticleInput && experimentArticleSuggestions) {
  let activeIndex = -1;
  let suggestions = [];
  let debounceTimer = null;

  const escapeHtml = (value) =>
    String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");

  const hideExperimentSuggestions = () => {
    suggestions = [];
    activeIndex = -1;
    experimentArticleSuggestions.hidden = true;
    experimentArticleSuggestions.innerHTML = "";
  };

  const applyExperimentSuggestion = (suggestion) => {
    experimentArticleInput.value = suggestion.title;
    if (experimentArticlePageId) {
      experimentArticlePageId.value = suggestion.page_id;
    }
    hideExperimentSuggestions();
  };

  const renderExperimentSuggestions = () => {
    if (!suggestions.length) {
      hideExperimentSuggestions();
      return;
    }

    experimentArticleSuggestions.innerHTML = suggestions
      .map(
        (item, index) => `
          <button
            type="button"
            class="suggestion-item ${index === activeIndex ? "active" : ""}"
            data-index="${index}"
          >
            <strong>${escapeHtml(item.title)}</strong>
            <span>Page ID ${escapeHtml(item.page_id)} | ${escapeHtml(item.file_name)}</span>
          </button>
        `
      )
      .join("");

    experimentArticleSuggestions.hidden = false;
    experimentArticleSuggestions.querySelectorAll(".suggestion-item").forEach((button) => {
      button.addEventListener("mousedown", (event) => {
        event.preventDefault();
        const index = Number(button.dataset.index);
        if (!Number.isNaN(index) && suggestions[index]) {
          applyExperimentSuggestion(suggestions[index]);
        }
      });
    });
  };

  const fetchExperimentSuggestions = async () => {
    const searchText = experimentArticleInput.value.trim();
    if (experimentArticlePageId) {
      experimentArticlePageId.value = "";
    }
    if (!searchText) {
      hideExperimentSuggestions();
      return;
    }

    const url = new URL(experimentArticleInput.dataset.suggestionsUrl, window.location.origin);
    url.searchParams.set("q", searchText);
    url.searchParams.set("limit", "10");

    const response = await fetch(url.toString(), {
      headers: { Accept: "application/json" },
    });
    if (!response.ok) {
      hideExperimentSuggestions();
      return;
    }

    const payload = await response.json();
    suggestions = Array.isArray(payload.suggestions) ? payload.suggestions : [];
    activeIndex = suggestions.length ? 0 : -1;
    renderExperimentSuggestions();
  };

  hideExperimentSuggestions();

  experimentArticleInput.addEventListener("input", () => {
    window.clearTimeout(debounceTimer);
    debounceTimer = window.setTimeout(fetchExperimentSuggestions, 120);
  });

  experimentArticleInput.addEventListener("keydown", (event) => {
    if (experimentArticleSuggestions.hidden || !suggestions.length) {
      return;
    }

    if (event.key === "ArrowDown") {
      event.preventDefault();
      activeIndex = (activeIndex + 1) % suggestions.length;
      renderExperimentSuggestions();
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      activeIndex = (activeIndex - 1 + suggestions.length) % suggestions.length;
      renderExperimentSuggestions();
    } else if (event.key === "Enter" && activeIndex >= 0) {
      event.preventDefault();
      applyExperimentSuggestion(suggestions[activeIndex]);
    } else if (event.key === "Escape") {
      hideExperimentSuggestions();
    }
  });

  document.addEventListener("click", (event) => {
    if (
      !experimentArticleSuggestions.contains(event.target) &&
      event.target !== experimentArticleInput
    ) {
      hideExperimentSuggestions();
    }
  });
}

if (experimentPickerForm && experimentPickerSubmit) {
  experimentPickerForm.addEventListener("submit", () => {
    experimentPickerSubmit.disabled = true;
    experimentPickerSubmit.textContent = "Loading editor...";
  });
}

if (experimentRunForm && experimentRunSubmit) {
  experimentRunForm.addEventListener("submit", () => {
    experimentRunSubmit.disabled = true;
    experimentRunSubmit.textContent = "Reranking...";
  });
}
