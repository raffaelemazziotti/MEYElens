import { registerDownloadLink } from "./asset-metrics.js";

(() => {
    const MODEL_ZOO_URL = "assets/json/model_zoo.json";

    async function loadModels() {
        const response = await fetch(MODEL_ZOO_URL);

        if (!response.ok) {
            throw new Error(`Could not load ${MODEL_ZOO_URL}`);
        }

        return response.json();
    }

    function hasValue(value) {
        if (value === null || value === undefined) return false;

        if (typeof value === "string" && value.trim() === "") {
            return false;
        }

        if (Array.isArray(value) && value.length === 0) {
            return false;
        }

        return true;
    }

    function escapeHtml(value) {
        return String(value ?? "")
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#039;");
    }

    function renderInlineMarkdown(value) {
        if (!hasValue(value)) return "";

        let html = escapeHtml(value);

        /*
         * Markdown links:
         * [text](https://example.com)
         */
        html = html.replace(
            /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
            '<a class="textlink" href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
        );

        /*
         * Bold:
         * **text**
         */
        html = html.replace(
            /\*\*([^*]+)\*\*/g,
            "<strong>$1</strong>"
        );

        /*
         * Italic:
         * *text*
         */
        html = html.replace(
            /(^|[\s(])\*([^*\n]+)\*/g,
            "$1<em>$2</em>"
        );

        /*
         * Inline code:
         * `text`
         */
        html = html.replace(
            /`([^`]+)`/g,
            "<code>$1</code>"
        );

        return html;
    }

    function setHtml(id, value) {
        const element = document.getElementById(id);

        if (!element) return;

        if (hasValue(value)) {
            element.innerHTML = renderInlineMarkdown(value);
            element.hidden = false;
        } else {
            element.innerHTML = "";
            element.hidden = true;
        }
    }

    function setText(id, value) {
        const element = document.getElementById(id);

        if (!element) return;

        if (hasValue(value)) {
            element.textContent = value;
            element.hidden = false;
        } else {
            element.textContent = "";
            element.hidden = true;
        }
    }

    function setLink(id, url) {
        const element = document.getElementById(id);

        if (!element) return;

        if (hasValue(url)) {
            element.href = url;
            element.hidden = false;
        } else {
            element.removeAttribute("href");
            element.hidden = true;
        }
    }

    function createModelCard(model) {
        const card = document.createElement("a");

        card.className = "model-zoo-card";
        card.href = `model-detail.html?id=${encodeURIComponent(model.id)}`;

        card.setAttribute(
            "aria-label",
            `Open details for ${model.name || "model"}`
        );

        const badgeHtml = hasValue(model.badge)
            ? `
                <span class="model-zoo-badge">
                    ${renderInlineMarkdown(model.badge)}
                </span>
            `
            : "";

        card.innerHTML = `
            <div class="model-zoo-thumb">
                <img
                    class="model-zoo-image"
                    src="${escapeHtml(model.image || "assets/media/model-placeholder.jpg")}"
                    alt="Preview of ${escapeHtml(model.name || "model")}"
                >
                ${badgeHtml}
            </div>

            <div class="model-zoo-body">
                ${
                    hasValue(model.tag)
                        ? `
                            <div class="model-zoo-type">
                                ${renderInlineMarkdown(model.tag)}
                            </div>
                        `
                        : ""
                }

                <h3>
                    ${renderInlineMarkdown(model.name || "Untitled model")}
                </h3>

                ${
                    hasValue(model.short_description)
                        ? `
                            <p>
                                ${renderInlineMarkdown(model.short_description)}
                            </p>
                        `
                        : ""
                }
            </div>
        `;

        return card;
    }

    function renderModelZoo(models) {
        const grid = document.getElementById("model-zoo-grid");
        const status = document.getElementById("model-zoo-status");

        if (!grid) return;

        grid.innerHTML = "";

        if (!Array.isArray(models) || !models.length) {
            if (status) {
                status.textContent = "No models available yet.";
            }

            return;
        }

        models.forEach((model) => {
            grid.appendChild(createModelCard(model));
        });

        if (status) {
            status.textContent = "";
        }
    }

    function addTableRow(tableBody, label, value) {
        if (!tableBody || !hasValue(value)) return;

        const row = document.createElement("tr");

        const th = document.createElement("th");
        const td = document.createElement("td");

        th.innerHTML = renderInlineMarkdown(label);

        if (Array.isArray(value)) {
            td.innerHTML = value
                .filter(hasValue)
                .map((item) => `<div>${renderInlineMarkdown(item)}</div>`)
                .join("");
        } else {
            td.innerHTML = renderInlineMarkdown(value);
        }

        row.appendChild(th);
        row.appendChild(td);
        tableBody.appendChild(row);
    }

    function renderObjectTable(tableId, data) {
        const tableBody = document.getElementById(tableId);

        if (!tableBody) return;

        tableBody.innerHTML = "";

        Object.entries(data || {}).forEach(([label, value]) => {
            addTableRow(tableBody, label, value);
        });
    }

    function renderSectionTable(sectionId, tableId, data) {
        const section = document.getElementById(sectionId);
        const tableBody = document.getElementById(tableId);

        if (!tableBody) return;

        tableBody.innerHTML = "";

        const entries = Object.entries(data || {}).filter(([, value]) => {
            return hasValue(value);
        });

        if (!entries.length) {
            if (section) {
                section.hidden = true;
            }

            return;
        }

        entries.forEach(([label, value]) => {
            addTableRow(tableBody, label, value);
        });

        if (section) {
            section.hidden = false;
        }
    }

    function renderOldDetailTables(model) {
        const specsTable = document.getElementById("model-specs-table");
        const performanceTable = document.getElementById("model-performance-table");

        if (specsTable) {
            specsTable.innerHTML = "";

            /*
             * Support both the newer nested specs object
             * and the older flat model fields.
             */
            if (hasValue(model.specs)) {
                Object.entries(model.specs).forEach(([label, value]) => {
                    addTableRow(specsTable, label, value);
                });
            } else {
                addTableRow(specsTable, "Backend", model.backend);
                addTableRow(specsTable, "Task", model.task);
                addTableRow(specsTable, "Input", model.input);
                addTableRow(specsTable, "Output", model.output);
                addTableRow(specsTable, "Recommended use", model.recommended_use);
                addTableRow(specsTable, "Package use", model.package_use);
            }
        }

        if (performanceTable) {
            renderObjectTable("model-performance-table", model.performance);
        }
    }

    function renderModelDetail(models) {
        const title = document.getElementById("model-title");

        if (!title) return;

        const params = new URLSearchParams(window.location.search);
        const modelId = params.get("id");

        const model = models.find((item) => item.id === modelId);

        if (!model) {
            title.textContent = "Model not found";

            const description = document.getElementById("model-description");

            if (description) {
                description.textContent =
                    "The requested model could not be found in the Model Zoo.";
            }

            return;
        }

        document.title = `MEYELens - ${model.name || "Model details"}`;

        /*
         * Main detail fields.
         * These support Markdown.
         */
        setHtml("model-title", model.name);
        setHtml("model-description", model.description);
        setHtml("model-tag", model.tag);
        setHtml("model-summary-name", model.name);
        setHtml("model-short-description", model.short_description);
        setHtml("model-notes", model.notes);
        setHtml("model-image-caption", model.image_caption);

        /*
         * Optional release field.
         */
        const modelRelease = document.getElementById("model-release");

        if (modelRelease) {
            if (hasValue(model.release)) {
                modelRelease.innerHTML =
                    `Release: ${renderInlineMarkdown(model.release)}`;
                modelRelease.hidden = false;
            } else {
                modelRelease.innerHTML = "";
                modelRelease.hidden = true;
            }
        }

        /*
         * Image.
         */
        const image = document.getElementById("model-image");

        if (image) {
            image.src = model.image || "assets/media/model-placeholder.jpg";
            image.alt = `Preview of ${model.name || "model"}`;
        }

        /*
         * Download links.
         */
        setLink("model-download-top", model.download);
        setLink("model-download-side", model.download);

        registerDownloadLink(
            document.getElementById("model-download-top"),
            "ai_model",
            model.id
        );
        registerDownloadLink(
            document.getElementById("model-download-side"),
            "ai_model",
            model.id
        );

        /*
         * New detail-page structure.
         */
        renderSectionTable(
            "model-specs-section",
            "model-specs-table",
            model.specs
        );

        renderSectionTable(
            "model-other-details-section",
            "model-other-details-table",
            model.other_details
        );

        /*
         * Old detail-page structure fallback.
         */
        renderOldDetailTables(model);
    }

    async function initModelZoo() {
        try {
            const models = await loadModels();

            renderModelZoo(models);
            renderModelDetail(models);
        } catch (error) {
            const status = document.getElementById("model-zoo-status");

            if (status) {
                status.textContent = "Could not load the Model Zoo.";
            }

            const title = document.getElementById("model-title");

            if (title) {
                title.textContent = "Could not load model details";
            }

            console.error(error);
        }
    }

    initModelZoo();
})();
