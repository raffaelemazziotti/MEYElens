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