const articleInput = document.querySelector("[data-article-input]");
const articleSuggestionsBox = document.querySelector("[data-article-suggestions]");
const articleSearchForm = document.querySelector("[data-article-search-form]");
const articleSubmitButton = document.querySelector("[data-article-submit-button]");

if (articleInput && articleSuggestionsBox) {
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
    articleSuggestionsBox.hidden = true;
    articleSuggestionsBox.innerHTML = "";
  };

  const openSuggestion = (suggestion) => {
    if (suggestion?.url) {
      window.location.assign(suggestion.url);
    }
  };

  const renderSuggestions = () => {
    if (!suggestions.length) {
      hideSuggestions();
      return;
    }

    articleSuggestionsBox.innerHTML = suggestions
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

    articleSuggestionsBox.hidden = false;
    articleSuggestionsBox.querySelectorAll(".suggestion-item").forEach((button) => {
      button.addEventListener("mousedown", (event) => {
        event.preventDefault();
        const index = Number(button.dataset.index);
        if (!Number.isNaN(index) && suggestions[index]) {
          openSuggestion(suggestions[index]);
        }
      });
    });
  };

  const fetchSuggestions = async () => {
    const searchText = articleInput.value.trim();
    if (!searchText) {
      hideSuggestions();
      return;
    }

    const url = new URL(articleInput.dataset.suggestionsUrl, window.location.origin);
    url.searchParams.set("q", searchText);
    url.searchParams.set("limit", "10");

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

  hideSuggestions();

  articleInput.addEventListener("input", () => {
    window.clearTimeout(debounceTimer);
    debounceTimer = window.setTimeout(fetchSuggestions, 120);
  });

  articleInput.addEventListener("keydown", (event) => {
    if (articleSuggestionsBox.hidden || !suggestions.length) {
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
      openSuggestion(suggestions[activeIndex]);
    } else if (event.key === "Escape") {
      hideSuggestions();
    }
  });

  document.addEventListener("click", (event) => {
    if (!articleSuggestionsBox.contains(event.target) && event.target !== articleInput) {
      hideSuggestions();
    }
  });
}

if (articleSearchForm && articleSubmitButton) {
  articleSearchForm.addEventListener("submit", () => {
    articleSubmitButton.disabled = true;
    articleSubmitButton.textContent = "Loading features...";
  });
}
