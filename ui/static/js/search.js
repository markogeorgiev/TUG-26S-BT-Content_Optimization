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
            <span>${escapeHtml(item.query_id)} | ${item.stored_result_count} rows${item.is_full_ranking ? "" : " | partial"}</span>
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

const sortableTable = document.querySelector("[data-sort-table]");

if (sortableTable) {
  const headerButtons = Array.from(sortableTable.querySelectorAll(".sort-button"));
  const tableBody = sortableTable.querySelector("tbody");

  const parseSortValue = (cell, type) => {
    const rawValue = cell?.dataset.sortValue ?? cell?.textContent?.trim() ?? "";
    if (type === "number") {
      const numericValue = Number(rawValue);
      return Number.isNaN(numericValue) ? Number.NEGATIVE_INFINITY : numericValue;
    }
    return String(rawValue).toLocaleLowerCase();
  };

  const refreshSortIndicators = (activeButton, direction) => {
    headerButtons.forEach((button) => {
      const isActive = button === activeButton;
      button.classList.toggle("is-active", isActive);
      button.dataset.sortDirection = isActive ? direction : "none";

      const indicator = button.querySelector(".sort-indicator");
      if (!indicator) {
        return;
      }

      if (!isActive) {
        indicator.textContent = "<>";
      } else if (direction === "asc") {
        indicator.textContent = "^";
      } else {
        indicator.textContent = "v";
      }
    });
  };

  headerButtons.forEach((button) => {
    button.addEventListener("click", () => {
      if (!tableBody) {
        return;
      }

      const direction = button.dataset.sortDirection === "asc" ? "desc" : "asc";
      const type = button.dataset.sortType || "string";
      const columnIndex = button.closest("th")?.cellIndex ?? 0;
      const sortableRows = Array.from(tableBody.querySelectorAll("tr")).map((row, originalIndex) => ({
        row,
        originalIndex,
        value: parseSortValue(row.cells[columnIndex], type),
      }));

      sortableRows.sort((left, right) => {
        let comparison = 0;
        if (type === "number") {
          comparison = left.value - right.value;
        } else {
          comparison = String(left.value).localeCompare(String(right.value), undefined, {
            numeric: true,
            sensitivity: "base",
          });
        }

        if (comparison === 0) {
          comparison = left.originalIndex - right.originalIndex;
        }

        return direction === "asc" ? comparison : -comparison;
      });

      tableBody.append(...sortableRows.map((item) => item.row));
      refreshSortIndicators(button, direction);
    });
  });
}
