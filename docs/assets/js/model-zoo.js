(() => {
    const MODEL_ZOO_URL = "assets/json/model_zoo.json";

    async function loadModels() {
        const response = await fetch(MODEL_ZOO_URL);

        if (!response.ok) {
            throw new Error(`Could not load ${MODEL_ZOO_URL}`);
        }

        return response.json();
    }

    function escapeHtml(value) {
        return String(value)
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#039;");
    }

    function renderInlineMarkdown(value) {
        if (!value) return "";

        let html = escapeHtml(value);

        // Markdown links: [text](https://example.com)
        html = html.replace(
            /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
            '<a class="textlink" href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
        );

        // Bold: **text**
        html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

        // Inline code: `text`
        html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

        return html;
    }

    function hasValue(value) {
        if (value === null || value === undefined) return false;
        if (typeof value === "string" && value.trim() === "") return false;
        if (Array.isArray(value) && value.length === 0) return false;
        return true;
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
        card.setAttribute("aria-label", `Open details for ${model.name}`);

        card.innerHTML = `
            <div class="model-zoo-thumb">
                <img
                    class="model-zoo-image"
                    src="${escapeHtml(model.image || "assets/media/model-placeholder.jpg")}"
                    alt="Preview of ${escapeHtml(model.name || "model")}"
                >
            </div>

            <div class="model-zoo-body">
                ${hasValue(model.tag) ? `<div class="model-zoo-type">${escapeHtml(model.tag)}</div>` : ""}
                <h3>${escapeHtml(model.name || "Untitled model")}</h3>
                ${hasValue(model.short_description) ? `<p>${escapeHtml(model.short_description)}</p>` : ""}
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
        if (!hasValue(value)) return;

        const row = document.createElement("tr");

        const th = document.createElement("th");
        th.textContent = label;

        const td = document.createElement("td");

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

    function renderTableSection(sectionId, tableId, data) {
        const section = document.getElementById(sectionId);
        const table = document.getElementById(tableId);

        if (!table) return;

        table.innerHTML = "";

        const entries = Object.entries(data || {}).filter(([, value]) => hasValue(value));

        if (!entries.length) {
            if (section) section.hidden = true;
            return;
        }

        entries.forEach(([label, value]) => {
            addTableRow(table, label, value);
        });

        if (section) section.hidden = false;
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
                description.textContent = "The requested model could not be found in the Model Zoo.";
            }

            return;
        }

        document.title = `MEYELens - ${model.name || "Model details"}`;

        setText("model-title", model.name);
        setText("model-description", model.description);
        setText("model-tag", model.tag);
        setText("model-summary-name", model.name);
        setText("model-short-description", model.short_description);

        const modelRelease = document.getElementById("model-release");
        if (modelRelease) {
            if (hasValue(model.release)) {
                modelRelease.textContent = `Release: ${model.release}`;
                modelRelease.hidden = false;
            } else {
                modelRelease.textContent = "";
                modelRelease.hidden = true;
            }
        }

        const image = document.getElementById("model-image");
        if (image) {
            image.src = model.image || "assets/media/model-placeholder.jpg";
            image.alt = `Preview of ${model.name || "model"}`;
        }

        setText("model-image-caption", model.image_caption);

        setLink("model-download-top", model.download);
        setLink("model-download-side", model.download);

        renderTableSection(
            "model-specs-section",
            "model-specs-table",
            model.specs
        );

        renderTableSection(
            "model-other-details-section",
            "model-other-details-table",
            model.other_details
        );
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