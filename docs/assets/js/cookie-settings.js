// cookie-settings.js
(() => {
    const consent = window.MEYELensConsent;
    const check = document.getElementById("consent-analytics");
    const status = document.getElementById("consent-status");

    if (!consent || !check) {
        return;
    }

    check.checked = consent.get() === "accepted";

    function leave(accepted) {
        consent.set(accepted);
        window.location.href = "index.html";
    }

    document.getElementById("consent-accept-all")?.addEventListener("click", () => leave(true));
    document.getElementById("consent-refuse-all")?.addEventListener("click", () => leave(false));

    document.getElementById("consent-save")?.addEventListener("click", () => {
        consent.set(check.checked);
        status.textContent = "Your preferences have been saved.";
    });
})();
