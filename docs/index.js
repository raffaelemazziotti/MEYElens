(() => {
    const yearEl = document.getElementById("year");
    if (yearEl) {
        yearEl.textContent = new Date().getFullYear();
    }

    const toggle = document.querySelector(".nav-toggle");
    const menu = document.getElementById("nav-menu");

    if (toggle && menu) {
        toggle.addEventListener("click", () => {
            const isOpen = menu.classList.toggle("open");
            toggle.setAttribute("aria-expanded", String(isOpen));
        });

        document.addEventListener("click", (event) => {
            const target = event.target;
            if (!menu.contains(target) && !toggle.contains(target)) {
                menu.classList.remove("open");
                toggle.setAttribute("aria-expanded", "false");
            }
        });
    }

    const updatesList = document.getElementById("updates-list");

    if (updatesList) {
        fetch("assets/updates.json")
            .then((response) => {
                if (!response.ok) {
                    throw new Error("Failed to load updates.json");
                }
                return response.json();
            })
            .then((updates) => {
                if (!Array.isArray(updates) || updates.length === 0) {
                    updatesList.innerHTML = "<p>No updates available yet.</p>";
                    return;
                }

                updates
                    .slice()
                    .sort((a, b) => new Date(b.date) - new Date(a.date))
                    .slice(0, 3)
                    .forEach((item) => {
                        const article = document.createElement("article");
                        article.className = "update-card";

                        const imageHtml = item.image
                            ? `
                                <div class="update-media">
                                    <img src="${item.image}" alt="${item.imageAlt || item.title || "Update image"}">
                                </div>
                            `
                            : "";

                        const tagHtml = item.tag
                            ? `<div class="update-tag">${item.tag}</div>`
                            : "";

                        const dateHtml = item.date
                            ? `<p class="update-date">${formatDate(item.date)}</p>`
                            : "";

                        const titleHtml = item.title
                            ? `<h3>${item.title}</h3>`
                            : "";

                        const summaryHtml = item.summary
                            ? `<p>${item.summary}</p>`
                            : "";

                        const linkHtml = item.url
                            ? `
                                <a class="btn" href="${item.url}">
                                    ${item.button || "Open"}
                                </a>
                            `
                            : "";

                        article.innerHTML = `
                            ${imageHtml}
                            <div class="update-body">
                                ${tagHtml}
                                ${dateHtml}
                                ${titleHtml}
                                ${summaryHtml}
                                ${linkHtml}
                            </div>
                        `;

                        updatesList.appendChild(article);
                    });
            })
            .catch((error) => {
                console.error("Error loading updates:", error);
                updatesList.innerHTML = "<p>Unable to load updates at the moment.</p>";
            });
    }

    function formatDate(dateString) {
        const date = new Date(dateString);

        if (Number.isNaN(date.getTime())) {
            return dateString;
        }

        return date.toLocaleDateString("en-GB", {
            day: "2-digit",
            month: "short",
            year: "numeric"
        });
    }
})();

document.querySelectorAll('.copy-btn').forEach(button => {
  button.addEventListener('click', () => {
    const codeBlock = button.closest('.code-block');
    const code = codeBlock ? codeBlock.querySelector('pre code') : null;

    if (!code) return;

    const text = code.innerText;

    navigator.clipboard.writeText(text).then(() => {
      button.textContent = 'Copied!';
      button.classList.add('copied');

      setTimeout(() => {
        button.textContent = 'Copy';
        button.classList.remove('copied');
      }, 1500);
    });
  });
});

function normalizeCodeBlocks() {
  document.querySelectorAll('pre code').forEach(block => {
    const lines = block.textContent
      .replace(/\t/g, '    ')
      .split('\n');

    while (lines.length && lines[0].trim() === '') {
      lines.shift();
    }

    while (lines.length && lines[lines.length - 1].trim() === '') {
      lines.pop();
    }

    const indents = lines
      .filter(line => line.trim())
      .map(line => {
        const match = line.match(/^ */);
        return match ? match[0].length : 0;
      });

    if (indents.length === 0) {
      block.textContent = '';
      return;
    }

    const minIndent = Math.min(...indents);

    block.textContent = lines
      .map(line => line.slice(minIndent))
      .join('\n');
  });
}

function setupCopyButtons() {
  document.querySelectorAll('.copy-btn').forEach(button => {
    button.addEventListener('click', () => {
      const codeBlock = button.closest('.code-block');
      const code = codeBlock ? codeBlock.querySelector('pre code') : null;

      if (!code) {
        return;
      }

      navigator.clipboard.writeText(code.innerText).then(() => {
        const originalText = button.textContent;

        button.textContent = 'Copied!';
        button.classList.add('copied');

        setTimeout(() => {
          button.textContent = originalText || 'Copy';
          button.classList.remove('copied');
        }, 1500);
      });
    });
  });
}

document.addEventListener('DOMContentLoaded', () => {
  normalizeCodeBlocks();

  if (window.Prism) {
    Prism.highlightAll();
  }

  setupCopyButtons();
});


/* installation matrix */

(() => {
    async function loadInstallConfig(matrix) {
        const configUrl = matrix.dataset.config || "install-options.json";
        const response = await fetch(configUrl);

        if (!response.ok) {
            throw new Error(`Could not load ${configUrl}`);
        }

        return response.json();
    }

    function createButton(group, option, isActive) {
        const button = document.createElement("button");

        button.type = "button";
        button.className = "install-cell";
        button.dataset.installGroup = group.id;
        button.dataset.installValue = option.value;
        button.textContent = option.label;
        button.setAttribute("aria-pressed", isActive ? "true" : "false");

        if (isActive) {
            button.classList.add("active");
        }

        return button;
    }

    function createGroupRow(group) {
        const row = document.createElement("div");
        row.className = "install-row";
        row.dataset.installGroup = group.id;

        const label = document.createElement("div");
        label.className = "install-row-label";
        label.textContent = group.label;

        const options = document.createElement("div");
        options.className = "install-row-options";
        options.style.setProperty("--install-columns", String(group.columns || group.options.length || 2));

        group.options.forEach((option, index) => {
            const defaultValue = group.default || group.options[0].value;
            const isActive = option.value === defaultValue || (!group.default && index === 0);
            options.appendChild(createButton(group, option, isActive));
        });

        row.appendChild(label);
        row.appendChild(options);

        return row;
    }

    function createOutputRow(config) {
    const row = document.createElement("div");
    row.className = "install-output-row";

    row.innerHTML = `
        <div class="install-row-label install-output-label">
            ${config.output_label || "Run this Command:"}
        </div>

        <div class="install-output">

            <div id="install-warning" class="install-gpu-warning" hidden>
                <strong id="install-warning-title"></strong>
                <p id="install-warning-text"></p>
                <a
                    id="install-warning-link"
                    class="textlink"
                    href="#"
                    target="_blank"
                    rel="noopener noreferrer"
                ></a>
                <p id="install-warning-small" class="install-warning-small"></p>
            </div>

            <div id="install-command-block" class="install-command-box">
                <div class="install-command-header">
                    <span>${config.command_label || "Install:"}</span>
                    <button class="copy-btn" type="button" aria-label="Copy code">Copy</button>
                </div>

                <pre><code id="install-command" class="language-bash"></code></pre>
            </div>

        </div>
    `;

    return row;
}

    function getState(matrix, config) {
        const state = {};

        config.groups.forEach((group) => {
            const activeButton = matrix.querySelector(
                `.install-cell.active[data-install-group="${group.id}"]`
            );

            state[group.id] = activeButton
                ? activeButton.dataset.installValue
                : group.default;
        });

        return state;
    }

    function getSelectedOptions(config, state) {
        return config.groups
            .map((group) => {
                const selectedValue = state[group.id];
                return group.options.find((option) => option.value === selectedValue);
            })
            .filter(Boolean);
    }

    function getExtras(config, state) {
        const selectedOptions = getSelectedOptions(config, state);

        const extras = selectedOptions
            .map((option) => option.extra)
            .filter((extra) => Boolean(extra));

        return extras;
    }

    function buildCommand(config, state) {
        const extras = getExtras(config, state);
        const extrasText = extras.length ? `[${extras.join(",")}]` : "";

        /*
            If there are extras:
            pip install "meyelens[tensorflow,gui]"

            If there are no extras:
            pip install meyelens
        */
        if (!extras.length) {
            return `pip install ${config.package || "meyelens"}`;
        }

        const template = config.default_command_template || "pip install \"{package}{extras}\"";

        return template
            .replaceAll("{package}", config.package || "meyelens")
            .replaceAll("{extras}", extrasText);
    }

    function findWarningKey(config, state) {
        const selectedOptions = getSelectedOptions(config, state);
        const warningOption = selectedOptions.find((option) => option.show_warning);

        if (!warningOption) {
            return null;
        }

        return warningOption.value;
    }

    function getWarning(config, state, warningKey) {
        const warningGroup = config.warnings && config.warnings[warningKey];

        if (!warningGroup) {
            return null;
        }

        if (state.backend && warningGroup[state.backend]) {
            return warningGroup[state.backend];
        }

        return warningGroup.default || null;
    }

    function updateOutput(matrix, config) {
    const state = getState(matrix, config);

    const commandBlock = matrix.querySelector("#install-command-block");
    const commandEl = matrix.querySelector("#install-command");

    const warningBox = matrix.querySelector("#install-warning");
    const warningTitle = matrix.querySelector("#install-warning-title");
    const warningText = matrix.querySelector("#install-warning-text");
    const warningLink = matrix.querySelector("#install-warning-link");
    const warningSmall = matrix.querySelector("#install-warning-small");

    const warningKey = findWarningKey(config, state);
    const warning = warningKey ? getWarning(config, state, warningKey) : null;

    commandBlock.hidden = false;
    commandEl.textContent = buildCommand(config, state);

    if (warning) {
        warningBox.hidden = false;

        warningTitle.textContent = warning.title || "";
        warningText.textContent = warning.text || "";
        warningLink.href = warning.link || "#";
        warningLink.textContent = warning.link_label || "Open instructions";
        warningSmall.textContent = warning.small_text || "";
    } else {
        warningBox.hidden = true;
    }

    if (window.Prism) {
        Prism.highlightElement(commandEl);
    }
}

    function bindButtons(matrix, config) {
        matrix.querySelectorAll(".install-cell").forEach((button) => {
            button.addEventListener("click", () => {
                const groupId = button.dataset.installGroup;

                matrix
                    .querySelectorAll(`.install-cell[data-install-group="${groupId}"]`)
                    .forEach((groupButton) => {
                        groupButton.classList.remove("active");
                        groupButton.setAttribute("aria-pressed", "false");
                    });

                button.classList.add("active");
                button.setAttribute("aria-pressed", "true");

                updateOutput(matrix, config);
            });
        });
    }

    function bindCopyButton(matrix) {
        const copyButton = matrix.querySelector(".copy-btn");
        const commandEl = matrix.querySelector("#install-command");

        if (!copyButton || !commandEl) return;

        copyButton.addEventListener("click", async () => {
            const command = commandEl.textContent.trim();

            try {
                await navigator.clipboard.writeText(command);
                copyButton.textContent = "Copied";
                copyButton.classList.add("copied");

                window.setTimeout(() => {
                    copyButton.textContent = "Copy";
                    copyButton.classList.remove("copied");
                }, 1400);
            } catch {
                copyButton.textContent = "Copy failed";

                window.setTimeout(() => {
                    copyButton.textContent = "Copy";
                }, 1400);
            }
        });
    }

    function renderInstallMatrix(matrix, config) {
        matrix.innerHTML = "";

        config.groups.forEach((group) => {
            matrix.appendChild(createGroupRow(group));
        });

        matrix.appendChild(createOutputRow(config));

        bindButtons(matrix, config);
        bindCopyButton(matrix);
        updateOutput(matrix, config);
    }

    async function initInstallMatrix() {
        const matrix = document.getElementById("meyelens-install");

        if (!matrix) return;

        try {
            const config = await loadInstallConfig(matrix);
            renderInstallMatrix(matrix, config);
        } catch (error) {
            matrix.innerHTML = `
                <div class="install-command-box">
                    <strong>Could not load installation options.</strong>
                    <p class="install-warning-small">
                        Check that the JSON file exists and that the page is served through a local or remote web server.
                    </p>
                </div>
            `;

            console.error(error);
        }
    }

    initInstallMatrix();
})();