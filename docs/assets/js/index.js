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
    const updatesToggle = document.getElementById("updates-toggle");

    if (updatesList) {
        let allUpdates = [];
        let updatesExpanded = false;

        const defaultVisibleUpdates = 3;

        fetch("assets/json/updates.json")
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

                allUpdates = updates
                    .slice()
                    .sort((a, b) => {
                        const dateA = new Date(`${a.date || ""}T00:00:00`);
                        const dateB = new Date(`${b.date || ""}T00:00:00`);

                        const timeA = Number.isNaN(dateA.getTime())
                            ? 0
                            : dateA.getTime();

                        const timeB = Number.isNaN(dateB.getTime())
                            ? 0
                            : dateB.getTime();

                        return timeB - timeA;
                    });

                renderUpdates();

                if (updatesToggle && allUpdates.length > defaultVisibleUpdates) {
                    updatesToggle.hidden = false;
                    updatesToggle.textContent = "Show all updates";
                    updatesToggle.setAttribute("aria-expanded", "false");

                    updatesToggle.addEventListener("click", () => {
                        updatesExpanded = !updatesExpanded;

                        updatesToggle.textContent = updatesExpanded
                            ? "Show fewer"
                            : "Show all updates";

                        updatesToggle.setAttribute(
                            "aria-expanded",
                            String(updatesExpanded)
                        );

                        renderUpdates();
                    });
                }
            })
            .catch((error) => {
                console.error("Error loading updates:", error);
                updatesList.innerHTML = "<p>Unable to load updates at the moment.</p>";
            });

        function renderUpdates() {
            updatesList.innerHTML = "";

            const visibleUpdates = updatesExpanded
                ? allUpdates
                : allUpdates.slice(0, defaultVisibleUpdates);

            visibleUpdates.forEach((item) => {
                const card = document.createElement(item.url ? "a" : "article");
                card.className = "update-card";

                if (item.url) {
                    card.href = item.url;
                    card.setAttribute("aria-label", item.title || "Open update");

                    if (isExternalUrl(item.url)) {
                        card.target = "_blank";
                        card.rel = "noopener noreferrer";
                    }
                }

                const badgeHtml = isRecentUpdate(item.date)
                    ? `<span class="update-badge">New</span>`
                    : "";

                const imageHtml = item.image
                    ? `
                        <div class="update-media">
                            <img
                                src="${escapeHtml(item.image)}"
                                alt="${escapeHtml(item.imageAlt || item.title || "Update image")}"
                            >
                            ${badgeHtml}
                        </div>
                    `
                    : "";

                const tagHtml = item.tag
                    ? `<div class="update-tag">${escapeHtml(item.tag)}</div>`
                    : "";

                const dateHtml = item.date
                    ? `<p class="update-date">${escapeHtml(formatDate(item.date))}</p>`
                    : "";

                const titleHtml = item.title
                    ? `<h3>${escapeHtml(item.title)}</h3>`
                    : "";

                const summaryHtml = item.summary
                    ? `<p>${escapeHtml(item.summary)}</p>`
                    : "";

                const buttonHtml = item.url
                    ? `
                        <span class="btn btn-primary update-card-button">
                            ${escapeHtml(item.button || "Open")}
                        </span>
                    `
                    : "";

                card.innerHTML = `
                    ${imageHtml}

                    <div class="update-body">
                        ${tagHtml}
                        ${dateHtml}
                        ${titleHtml}
                        ${summaryHtml}
                        ${buttonHtml}
                    </div>
                `;

                updatesList.appendChild(card);
            });
        }
    }

    function formatDate(dateString) {
        const date = new Date(`${dateString}T00:00:00`);

        if (Number.isNaN(date.getTime())) {
            return dateString;
        }

        return date.toLocaleDateString("en-GB", {
            day: "2-digit",
            month: "short",
            year: "numeric"
        });
    }

    function isRecentUpdate(dateString, maxAgeDays = 30) {
        if (!dateString) return false;

        const updateDate = new Date(`${dateString}T00:00:00`);

        if (Number.isNaN(updateDate.getTime())) {
            return false;
        }

        const today = new Date();

        today.setHours(0, 0, 0, 0);
        updateDate.setHours(0, 0, 0, 0);

        const ageMs = today.getTime() - updateDate.getTime();
        const maxAgeMs = maxAgeDays * 24 * 60 * 60 * 1000;

        return ageMs >= 0 && ageMs <= maxAgeMs;
    }

    function isExternalUrl(url) {
        return /^https?:\/\//i.test(url);
    }

    function escapeHtml(value) {
        return String(value ?? "")
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#039;");
    }
})();

function normalizeCodeBlocks() {
    document.querySelectorAll("pre code").forEach((block) => {
        const lines = block.textContent
            .replace(/\t/g, "    ")
            .split("\n");

        while (lines.length && lines[0].trim() === "") {
            lines.shift();
        }

        while (lines.length && lines[lines.length - 1].trim() === "") {
            lines.pop();
        }

        const indents = lines
            .filter((line) => line.trim())
            .map((line) => {
                const match = line.match(/^ */);
                return match ? match[0].length : 0;
            });

        if (indents.length === 0) {
            block.textContent = "";
            return;
        }

        const minIndent = Math.min(...indents);

        block.textContent = lines
            .map((line) => line.slice(minIndent))
            .join("\n");
    });
}

function setupCopyButtons() {
    document.querySelectorAll(".copy-btn").forEach((button) => {
        button.addEventListener("click", () => {
            const codeBlock = button.closest(".code-block");
            const code = codeBlock ? codeBlock.querySelector("pre code") : null;

            if (!code) {
                return;
            }

            navigator.clipboard.writeText(code.innerText).then(() => {
                const originalText = button.textContent;

                button.textContent = "Copied!";
                button.classList.add("copied");

                setTimeout(() => {
                    button.textContent = originalText || "Copy";
                    button.classList.remove("copied");
                }, 1500);
            });
        });
    });
}

document.addEventListener("DOMContentLoaded", () => {
    normalizeCodeBlocks();

    if (window.Prism) {
        Prism.highlightAll();
    }

    setupCopyButtons();
});


/* installation matrix */

(() => {
    async function loadInstallConfig(matrix) {
        const configUrl = matrix.dataset.config || "assets/json/install-options.json";
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
        options.style.setProperty(
            "--install-columns",
            String(group.columns || group.options.length || 2)
        );

        group.options.forEach((option, index) => {
            const defaultValue = group.default || group.options[0].value;
            const isActive =
                option.value === defaultValue ||
                (!group.default && index === 0);

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

                <div id="install-warning" class="install-gpu-warning" hidden></div>

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

                return group.options.find(
                    (option) => option.value === selectedValue
                );
            })
            .filter(Boolean);
    }

    function getExtras(config, state) {
        const selectedOptions = getSelectedOptions(config, state);

        return selectedOptions
            .map((option) => option.extra)
            .filter((extra) => Boolean(extra));
    }

    function getPackageName(config, state) {
        const selectedOptions = getSelectedOptions(config, state);

        const packageOption = selectedOptions.find(
            (option) => option.package
        );

        return packageOption
            ? packageOption.package
            : config.package || "meyelens";
    }

    function buildCommand(config, state) {
        const packageName = getPackageName(config, state);
        const extras = getExtras(config, state);
        const extrasText = extras.length ? `[${extras.join(",")}]` : "";

        if (!extras.length) {
            return `pip install ${packageName}`;
        }

        const template =
            config.default_command_template ||
            "pip install \"{package}{extras}\"";

        return template
            .replaceAll("{package}", packageName)
            .replaceAll("{extras}", extrasText);
    }

    function getWarnings(config, state) {
        const selectedOptions = getSelectedOptions(config, state);

        return selectedOptions
            .filter((option) => option.show_warning)
            .map((option) => {
                const warningGroup =
                    config.warnings && config.warnings[option.value];

                if (!warningGroup) {
                    return null;
                }

                return warningGroup.default || null;
            })
            .filter(Boolean);
    }

    function updateOutput(matrix, config) {
        const state = getState(matrix, config);

        const commandBlock = matrix.querySelector("#install-command-block");
        const commandEl = matrix.querySelector("#install-command");
        const warningBox = matrix.querySelector("#install-warning");

        const warnings = getWarnings(config, state);

        if (commandBlock) {
            commandBlock.hidden = false;
        }

        if (commandEl) {
            commandEl.textContent = buildCommand(config, state);
        }

        if (warningBox) {
            if (warnings.length) {
                warningBox.hidden = false;

                warningBox.innerHTML = warnings
                    .map((warning) => `
                        <div class="install-warning-item">
                            <strong>${warning.title || ""}</strong>
                            <p>${warning.text || ""}</p>

                            ${
                                warning.link
                                    ? `
                                        <a
                                            class="textlink"
                                            href="${warning.link}"
                                            target="_blank"
                                            rel="noopener noreferrer"
                                        >
                                            ${warning.link_label || "Open instructions"}
                                        </a>
                                    `
                                    : ""
                            }

                            <p class="install-warning-small">
                                ${warning.small_text || ""}
                            </p>
                        </div>
                    `)
                    .join("");
            } else {
                warningBox.hidden = true;
                warningBox.innerHTML = "";
            }
        }

        if (window.Prism && commandEl) {
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

function addSiteNotice() {
    const existingNotice = document.querySelector(".site-notice");

    if (existingNotice) {
        return;
    }

    const header = document.querySelector(".site-header");

    if (!header) {
        return;
    }

    const notice = document.createElement("div");
    notice.className = "site-notice";

    notice.innerHTML = `
        <div class="container">
            <strong>Work in progress.</strong>
            This website is being actively updated as
            <span class="brand-accent">MEYE</span><span class="brand-normal">Lens</span>
            documentation, tutorials, and resources are completed.
        </div>
    `;

    header.insertAdjacentElement("afterend", notice);
}

//addSiteNotice();