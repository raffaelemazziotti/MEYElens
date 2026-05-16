(() => {
    const MODEL_ZOO_URL = "model_zoo.json";

    async function loadModels() {
        const response = await fetch(MODEL_ZOO_URL);

        if (!response.ok) {
            throw new Error(`Could not load ${MODEL_ZOO_URL}`);
        }

        return response.json();
    }

    function createModelCard(model) {
        const article = document.createElement("article");
        article.className = "model-zoo-card";

        article.innerHTML = `
            <img
                class="model-zoo-image"
                src="${model.image}"
                alt="Preview of ${model.name}"
            >

            <div class="model-zoo-body">
                <div class="resource-type">${model.tag}</div>
                <h3>${model.name}</h3>
                <p>${model.short_description}</p>

                <div class="model-zoo-actions">
                    <a class="btn btn-primary" href="${model.download}" download>
                        Download
                    </a>
                    <a class="btn" href="model-detail.html?id=${encodeURIComponent(model.id)}">
                        Info
                    </a>
                </div>
            </div>
        `;

        return article;
    }

    function renderModelZoo(models) {
        const grid = document.getElementById("model-zoo-grid");
        const status = document.getElementById("model-zoo-status");

        if (!grid) return;

        grid.innerHTML = "";

        if (!models.length) {
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
        const row = document.createElement("tr");

        const th = document.createElement("th");
        th.textContent = label;

        const td = document.createElement("td");
        td.textContent = value || "To be added";

        row.appendChild(th);
        row.appendChild(td);
        tableBody.appendChild(row);
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

        document.title = `MEYELens - ${model.name}`;

        const modelTitle = document.getElementById("model-title");
        const modelDescription = document.getElementById("model-description");
        const modelTag = document.getElementById("model-tag");
        const modelSummaryName = document.getElementById("model-summary-name");
        const modelShortDescription = document.getElementById("model-short-description");
        const modelNotes = document.getElementById("model-notes");

        if (modelTitle) modelTitle.textContent = model.name;
        if (modelDescription) modelDescription.textContent = model.description;
        if (modelTag) modelTag.textContent = model.tag;
        if (modelSummaryName) modelSummaryName.textContent = model.name;
        if (modelShortDescription) modelShortDescription.textContent = model.short_description;

        /*
            Notes can contain trusted HTML from model_zoo.json.

            Example JSON:
            "notes": "Released in 2022-01-24 - <a class=\"textlink\" href=\"https://github.com/fabiocarrara/meye/releases/tag/v0.1.1\" target=\"_blank\" rel=\"noopener noreferrer\">Reference</a>"
        */
        if (modelNotes) {
            modelNotes.innerHTML = model.notes || "";
        }

        const image = document.getElementById("model-image");
        if (image) {
            image.src = model.image;
            image.alt = `Preview of ${model.name}`;
        }

        const caption = document.getElementById("model-image-caption");
        if (caption) {
            caption.textContent = model.image_caption || "Model preview.";
        }

        const downloadTop = document.getElementById("model-download-top");
        const downloadSide = document.getElementById("model-download-side");

        if (downloadTop) {
            downloadTop.href = model.download;
        }

        if (downloadSide) {
            downloadSide.href = model.download;
        }

        const specsTable = document.getElementById("model-specs-table");
        const performanceTable = document.getElementById("model-performance-table");

        if (specsTable) {
            specsTable.innerHTML = "";

            addTableRow(specsTable, "Backend", model.backend);
            addTableRow(specsTable, "Task", model.task);
            addTableRow(specsTable, "Input", model.input);
            addTableRow(specsTable, "Output", model.output);
            addTableRow(specsTable, "Recommended use", model.recommended_use);
            addTableRow(specsTable, "Package use", model.package_use);
        }

        if (performanceTable) {
            performanceTable.innerHTML = "";

            Object.entries(model.performance || {}).forEach(([label, value]) => {
                addTableRow(performanceTable, label, value);
            });
        }
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