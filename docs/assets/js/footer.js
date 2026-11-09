// footer.js
// Single source for the site footer. It must run after nav.js (the header comes
// first on every page): the footer links are copied from the rendered main menu,
// so the current-page highlight and the link list stay in sync with it.
(() => {
    const footer = document.getElementById("site-footer");
    if (!footer) return;

    const menuLinks = [...document.querySelectorAll("#nav-menu a")]
        .map((link) => link.outerHTML)
        .join("\n            ");

    footer.innerHTML = `
    <div class="container footer-inner">
        <div>
            <div class="footer-brand"><span class="brand-accent">MEYE</span><span class="brand-normal">Lens</span></div>
            <span class="brand-tag">
                Released under <a href="https://www.gnu.org/licenses/gpl-3.0.en.html" target="_blank" rel="noopener noreferrer" class="textlink">GPL-3.0</a> (code)
                and <a href="https://spdx.org/licenses/CERN-OHL-P-2.0.html" target="_blank" rel="noopener noreferrer" class="textlink">CERN-OHL-P-2.0</a> (hardware).
            </span>
        </div>

        <div class="footer-links">
            ${menuLinks}
            <a href="cookie-settings.html">Cookie settings</a>
        </div>
    </div>`;
})();
