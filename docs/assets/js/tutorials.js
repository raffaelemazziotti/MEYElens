import { recordAssetMetric } from "./asset-metrics.js";

(() => {
    const TUTORIALS_URL = "assets/json/tutorials.json";

    async function loadTutorials() {
        const response = await fetch(TUTORIALS_URL);

        if (!response.ok) {
            throw new Error(`Could not load ${TUTORIALS_URL}`);
        }

        return response.json();
    }

    async function loadTextFile(url) {
        const response = await fetch(url);

        if (!response.ok) {
            throw new Error(`Could not load ${url}`);
        }

        return response.text();
    }

    function hasValue(value) {
        if (value === null || value === undefined) return false;
        if (typeof value === "string" && value.trim() === "") return false;
        if (Array.isArray(value) && value.length === 0) return false;
        return true;
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

        html = html.replace(
            /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
            '<a class="textlink" href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
        );

        html = html.replace(
            /\[([^\]]+)\]\(([^)\s]+)\)/g,
            '<a class="textlink" href="$2">$1</a>'
        );

        html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
        html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

        return html;
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

    function createTutorialCard(tutorial) {
        const card = document.createElement("a");
        card.className = "tutorial-card";
        card.href = `tutorial-detail.html?id=${encodeURIComponent(tutorial.id)}`;
        card.setAttribute("aria-label", `Open tutorial: ${tutorial.title}`);

        card.innerHTML = `
            <div class="tutorial-thumb">
                <img
                    src="${escapeHtml(tutorial.image || "assets/media/favicon.png")}"
                    alt="${escapeHtml(tutorial.imageAlt || tutorial.title || "Tutorial preview")}"
                >
            </div>
            <div class="tutorial-body">
                ${hasValue(tutorial.tag) ? `<div class="tutorial-type">${escapeHtml(tutorial.tag)}</div>` : ""}
                <h3>${escapeHtml(tutorial.title || "Untitled tutorial")}</h3>
                ${hasValue(tutorial.summary) ? `<p>${escapeHtml(tutorial.summary)}</p>` : ""}
            </div>
        `;

        return card;
    }

    function renderTutorialGallery(tutorials) {
        const gallery = document.getElementById("tutorial-gallery");
        const status = document.getElementById("tutorial-status");

        if (!gallery) return;

        gallery.innerHTML = "";

        if (!Array.isArray(tutorials) || tutorials.length === 0) {
            if (status) {
                status.textContent = "No tutorials available yet.";
            }
            return;
        }

        tutorials.forEach((tutorial) => {
            gallery.appendChild(createTutorialCard(tutorial));
        });

        if (status) {
            status.textContent = "";
        }
    }

    function createTextSection(section) {
        const block = document.createElement("div");
        block.className = "text-block tutorial-section";

        block.innerHTML = `
            ${hasValue(section.title) ? `<h2>${escapeHtml(section.title)}</h2>` : ""}
            ${hasValue(section.body) ? `<p>${renderInlineMarkdown(section.body)}</p>` : ""}
        `;

        return block;
    }

    function createNoteSection(section, noteClass) {
        const block = document.createElement("div");
        block.className = `note ${noteClass} tutorial-section`;

        const label = section.title || section.kind || "Note";

        block.innerHTML = `
            <strong>${escapeHtml(label)}:</strong>
            ${hasValue(section.body) ? renderInlineMarkdown(section.body) : ""}
        `;

        return block;
    }

    function createListSection(section) {
        const block = document.createElement("div");
        block.className = "content-block tutorial-section";

        const items = Array.isArray(section.items) ? section.items : [];

        block.innerHTML = `
            ${hasValue(section.title) ? `<h2>${escapeHtml(section.title)}</h2>` : ""}
            <ul class="list">
                ${items
                    .filter(hasValue)
                    .map((item) => `<li>${renderInlineMarkdown(item)}</li>`)
                    .join("")}
            </ul>
        `;

        return block;
    }

    function createImageSection(section) {
        const figure = document.createElement("figure");
        figure.className = "media-figure tutorial-section";

        figure.innerHTML = `
            <img src="${escapeHtml(section.src || "")}" alt="${escapeHtml(section.alt || section.title || "")}">
            ${hasValue(section.caption) ? `<figcaption>${renderInlineMarkdown(section.caption)}</figcaption>` : ""}
        `;

        return figure;
    }

    async function createCodeSection(section, tutorial) {
        const source =
            section.source === "python_file"
                ? tutorial.python_file
                : section.source;

        const code = source ? await loadTextFile(source) : "";

        const block = document.createElement("div");
        block.className = "code-block tutorial-section";

        block.innerHTML = `
            <div class="code-header">
                <span class="code-label">${escapeHtml(section.title || "Python")}</span>
                <button class="copy-btn" type="button" aria-label="Copy code">Copy</button>
            </div>
            <pre><code class="language-python">${escapeHtml(code)}</code></pre>
        `;

        return block;
    }

    function createLinkSection(section) {
        const block = document.createElement("div");
        block.className = "content-block tutorial-section";

        block.innerHTML = `
            ${hasValue(section.title) ? `<h2>${escapeHtml(section.title)}</h2>` : ""}
            ${hasValue(section.body) ? `<p>${renderInlineMarkdown(section.body)}</p>` : ""}
            ${hasValue(section.url) ? `<a class="btn" href="${escapeHtml(section.url)}">${escapeHtml(section.button || "Open")}</a>` : ""}
        `;

        return block;
    }

    async function createTutorialSection(section, tutorial) {
        const kind = section.kind || "text";

        if (kind === "tip") {
            return createNoteSection(section, "note-tip");
        }

        if (kind === "warning") {
            return createNoteSection(section, "note-warning");
        }

        if (kind === "note") {
            return createNoteSection(section, "note-tip");
        }

        if (kind === "list") {
            return createListSection(section);
        }

        if (kind === "image") {
            return createImageSection(section);
        }

        if (kind === "code") {
            return createCodeSection(section, tutorial);
        }

        if (kind === "link") {
            return createLinkSection(section);
        }

        return createTextSection(section);
    }

    function renderTutorialPreview(tutorial) {
        const previewCard = document.getElementById("tutorial-preview-card");
        const image = document.getElementById("tutorial-image");
        const imageCaption = document.getElementById("tutorial-image-caption");

        if (!previewCard || !image) return;

        if (!hasValue(tutorial.image)) {
            previewCard.hidden = true;
            return;
        }

        image.src = tutorial.image;
        image.alt = tutorial.imageAlt || tutorial.title || "Tutorial image";

        if (imageCaption) {
            imageCaption.innerHTML = hasValue(tutorial.imageCaption)
                ? renderInlineMarkdown(tutorial.imageCaption)
                : renderInlineMarkdown(tutorial.summary || "");
        }

        previewCard.hidden = false;
    }

    function renderTutorialMeta(tutorial) {
        const meta = document.getElementById("tutorial-meta");

        if (!meta) return;

        const items = [
            tutorial.tag ? `Type: ${tutorial.tag}` : "",
            tutorial.level ? `Level: ${tutorial.level}` : "",
            //tutorial.python_file ? "Includes Python script" : ""
        ].filter(hasValue);

        if (!items.length) {
            meta.innerHTML = "";
            return;
        }

        meta.innerHTML = items
            .map((item) => `<div class="tutorial-pill">${escapeHtml(item)}</div>`)
            .join("");
    }

    async function renderTutorialDetail(tutorials) {
        const titleElement = document.getElementById("tutorial-title");

        if (!titleElement) return;

        const params = new URLSearchParams(window.location.search);
        const tutorialId = params.get("id");
        const tutorial = tutorials.find((item) => item.id === tutorialId);

        if (!tutorial) {
            titleElement.textContent = "Tutorial not found";

            const description = document.getElementById("tutorial-description");
            if (description) {
                description.textContent = "The requested tutorial could not be found.";
            }

            return;
        }

        document.title = `MEYELens - ${tutorial.title || "Tutorial"}`;

        // A tutorial has no downloadable asset at present, so this records an
        // actual visit to its detail page instead of a fictitious download.
        void recordAssetMetric("tutorial", tutorial.id, "view");

        setText("tutorial-tag", tutorial.tag || "Tutorial");
        setText("tutorial-title", tutorial.title);
        setText("tutorial-description", tutorial.description);
        setText("tutorial-level", tutorial.level || tutorial.tag);

        renderTutorialPreview(tutorial);
        renderTutorialMeta(tutorial);

        const content = document.getElementById("tutorial-content");
        if (!content) return;

        content.innerHTML = "";

        const sections = Array.isArray(tutorial.sections) ? tutorial.sections : [];

        for (const section of sections) {
            const element = await createTutorialSection(section, tutorial);
            content.appendChild(element);
        }

        if (window.Prism) {
            window.Prism.highlightAll();
        }
    }

    async function initTutorials() {
        try {
            const tutorials = await loadTutorials();

            renderTutorialGallery(tutorials);
            await renderTutorialDetail(tutorials);
        } catch (error) {
            const status = document.getElementById("tutorial-status");
            if (status) {
                status.textContent = "Could not load tutorials.";
            }

            const title = document.getElementById("tutorial-title");
            if (title) {
                title.textContent = "Could not load tutorial";
            }

            console.error(error);
        }
    }

    initTutorials();
})();
