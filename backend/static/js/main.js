// Small JavaScript helpers for navigation and subtle UI enhancements.

document.addEventListener("DOMContentLoaded", () => {
  const navToggle = document.getElementById("navToggle");
  const nav = document.querySelector(".nav");
  const speciesSelect = document.getElementById("species");
  const speciesPreview = document.getElementById("speciesPreview");

  if (navToggle && nav) {
    navToggle.addEventListener("click", () => {
      nav.classList.toggle("open");
    });
  }

  // Show a small icon + label preview when a pet type is selected.
  if (speciesSelect && speciesPreview) {
    const iconSpan = speciesPreview.querySelector(".species-icon");
    const labelSpan = speciesPreview.querySelector(".species-label");

    const updatePreview = () => {
      const option = speciesSelect.options[speciesSelect.selectedIndex];
      if (!option || !option.value) {
        iconSpan.textContent = "🐾";
        labelSpan.textContent = "Choose a pet type to see its icon.";
        return;
      }
      const icon = option.getAttribute("data-icon") || "🐾";
      iconSpan.textContent = icon;
      labelSpan.textContent = option.textContent.trim();
    };

    speciesSelect.addEventListener("change", updatePreview);
    // Initialize on page load if a value is pre-selected.
    updatePreview();
  }
});

