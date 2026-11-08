// nav.js
// Single source for the main navigation. Loaded synchronously right after the
// empty #nav-menu element so the links exist before the page is painted.
(() => {
    const menu = document.getElementById("nav-menu");
    if (!menu) return;

    const ITEMS = [
        { href: "index.html", label: "Home" },
        { href: "get_started.html", label: "Get Started" },
        { href: "hardware.html", label: "Hardware" },
        { href: "software.html", label: "Software" },
        { href: "models.html", label: "3D Models" },
        { href: "resources.html", label: "Resources" },
        { href: "contact.html", label: "Contact" }
    ];

    // Detail pages highlight the section they belong to.
    const SECTION_OF = {
        "model-detail.html": "software.html",
        "tutorial-detail.html": "resources.html"
    };

    function currentPage() {
        let page = window.location.pathname.split("/").pop() || "index.html";
        if (!page.includes(".")) page += ".html";
        return SECTION_OF[page] || page;
    }

    const current = currentPage();

    menu.innerHTML = ITEMS.map(({ href, label }) => (
        href === current
            ? `<a href="${href}" aria-current="page"><span class="brand-accent">${label}</span></a>`
            : `<a href="${href}">${label}</a>`
    )).join("\n");

    // Mobile: the toggle itself is wired in index.js. Also close the menu when
    // a link is chosen, when Escape is pressed, or when the layout goes desktop.
    const toggle = document.querySelector(".nav-toggle");

    function closeMenu() {
        menu.classList.remove("open");
        toggle?.setAttribute("aria-expanded", "false");
    }

    menu.addEventListener("click", (event) => {
        if (event.target.closest("a")) closeMenu();
    });

    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && menu.classList.contains("open")) {
            closeMenu();
            toggle?.focus();
        }
    });

    window.matchMedia("(min-width: 761px)").addEventListener("change", (event) => {
        if (event.matches) closeMenu();
    });
})();
