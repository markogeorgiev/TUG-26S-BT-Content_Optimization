const queryInput = document.querySelector("[data-query-input]");
const suggestionsBox = document.querySelector("[data-suggestions]");
const searchForm = document.querySelector("[data-search-form]");
const submitButton = document.querySelector("[data-submit-button]");
const displayLimitSelect = document.querySelector("[data-display-limit-select]");
const customLimitBox = document.querySelector("[data-custom-limit-box]");

if (queryInput && suggestionsBox) {
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

  const hideSuggestions = () => {
    suggestions = [];
    activeIndex = -1;
    suggestionsBox.hidden = true;
    suggestionsBox.innerHTML = "";
  };

  hideSuggestions();

  const applySuggestion = (suggestion) => {
    queryInput.value = suggestion.query_text;
    hideSuggestions();
  };

  const renderSuggestions = () => {
    if (!suggestions.length) {
      hideSuggestions();
      return;
    }

    suggestionsBox.innerHTML = suggestions
      .map(
        (item, index) => `
          <button
            type="button"
            class="suggestion-item ${index === activeIndex ? "active" : ""}"
            data-index="${index}"
          >
            <strong>${escapeHtml(item.query_text)}</strong>
            <span>${escapeHtml(item.query_id)} · ${item.stored_result_count} rows${item.is_full_ranking ? "" : " · partial"}</span>
          </button>
        `
      )
      .join("");

    suggestionsBox.hidden = false;
    suggestionsBox.querySelectorAll(".suggestion-item").forEach((button) => {
      button.addEventListener("mousedown", (event) => {
        event.preventDefault();
        const index = Number(button.dataset.index);
        if (!Number.isNaN(index) && suggestions[index]) {
          applySuggestion(suggestions[index]);
        }
      });
    });
  };

  const fetchSuggestions = async () => {
    const query = queryInput.value.trim();
    if (!query) {
      hideSuggestions();
      return;
    }

    const url = new URL(queryInput.dataset.suggestionsUrl, window.location.origin);
    url.searchParams.set("q", query);
    url.searchParams.set("limit", "8");

    const response = await fetch(url.toString(), {
      headers: { Accept: "application/json" },
    });
    if (!response.ok) {
      hideSuggestions();
      return;
    }

    const payload = await response.json();
    suggestions = Array.isArray(payload.suggestions) ? payload.suggestions : [];
    activeIndex = suggestions.length ? 0 : -1;
    renderSuggestions();
  };

  queryInput.addEventListener("input", () => {
    window.clearTimeout(debounceTimer);
    debounceTimer = window.setTimeout(fetchSuggestions, 120);
  });

  queryInput.addEventListener("keydown", (event) => {
    if (suggestionsBox.hidden || !suggestions.length) {
      return;
    }

    if (event.key === "ArrowDown") {
      event.preventDefault();
      activeIndex = (activeIndex + 1) % suggestions.length;
      renderSuggestions();
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      activeIndex = (activeIndex - 1 + suggestions.length) % suggestions.length;
      renderSuggestions();
    } else if (event.key === "Enter" && activeIndex >= 0) {
      event.preventDefault();
      applySuggestion(suggestions[activeIndex]);
      searchForm.requestSubmit();
    } else if (event.key === "Escape") {
      hideSuggestions();
    }
  });

  document.addEventListener("click", (event) => {
    if (!suggestionsBox.contains(event.target) && event.target !== queryInput) {
      hideSuggestions();
    }
  });
}

if (displayLimitSelect && customLimitBox) {
  const syncCustomLimitVisibility = () => {
    const shouldShowCustom = displayLimitSelect.value === "custom";
    customLimitBox.classList.toggle("is-hidden", !shouldShowCustom);
  };

  syncCustomLimitVisibility();
  displayLimitSelect.addEventListener("change", syncCustomLimitVisibility);
}

if (searchForm && submitButton) {
  searchForm.addEventListener("submit", () => {
    submitButton.disabled = true;
    submitButton.textContent = "Running query...";
  });
}
