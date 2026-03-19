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
})();