// analytics-consent.js
(() => {
    const MEASUREMENT_ID = "G-BZ0DN84FBW";
    const STORAGE_KEY = "meyelens_cookie_consent";
    const PRIVACY_URL = "privacy.html";
    const SETTINGS_URL = "cookie-settings.html";
    const NO_BANNER_PAGES = ["/privacy", "/cookie-settings"];

    function getStoredConsent() {
        return localStorage.getItem(STORAGE_KEY);
    }

    function setStoredConsent(value) {
        localStorage.setItem(STORAGE_KEY, value);
    }

    function ensureDataLayer() {
        window.dataLayer = window.dataLayer || [];
        window.gtag = window.gtag || function gtag() {
            window.dataLayer.push(arguments);
        };
    }

    function loadGtagScript() {
        if (document.querySelector(`script[data-ga="${MEASUREMENT_ID}"]`)) {
            return;
        }

        const script = document.createElement("script");
        script.async = true;
        script.src = `https://www.googletagmanager.com/gtag/js?id=${MEASUREMENT_ID}`;
        script.dataset.ga = MEASUREMENT_ID;
        document.head.appendChild(script);
    }

    function enableAnalytics() {
        ensureDataLayer();
        loadGtagScript();

        window[`ga-disable-${MEASUREMENT_ID}`] = false;
        window.gtag("js", new Date());

        // Explicitly mark consent as granted before config
        window.gtag("consent", "update", {
            analytics_storage: "granted",
            ad_storage: "denied",
            ad_user_data: "denied",
            ad_personalization: "denied"
        });

        window.gtag("config", MEASUREMENT_ID, {
            anonymize_ip: true
        });
    }

    function disableAnalytics() {
        ensureDataLayer();

        // Stop any running tag and keep consent denied
        window[`ga-disable-${MEASUREMENT_ID}`] = true;
        window.gtag("consent", "update", {
            analytics_storage: "denied",
            ad_storage: "denied",
            ad_user_data: "denied",
            ad_personalization: "denied"
        });
        clearAnalyticsCookies();
    }

    function clearAnalyticsCookies() {
        const names = ["_ga", "_gid", `_ga_${MEASUREMENT_ID.replace("G-", "")}`];
        const host = window.location.hostname;
        const domains = [host, `.${host}`];
        const parts = host.split(".");
        if (parts.length > 2) {
            domains.push(`.${parts.slice(-2).join(".")}`);
        }
        names.forEach((name) => {
            document.cookie = `${name}=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/`;
            domains.forEach((domain) => {
                document.cookie = `${name}=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/; domain=${domain}`;
            });
        });
    }

    function setConsent(accepted) {
        setStoredConsent(accepted ? "accepted" : "declined");
        if (accepted) {
            enableAnalytics();
        } else {
            disableAnalytics();
        }
        removeBanner();
    }

    function injectSettingsLink() {
        const links = document.querySelector(".footer-links");
        if (!links || links.querySelector(`a[href="${SETTINGS_URL}"]`)) {
            return;
        }
        const link = document.createElement("a");
        link.href = SETTINGS_URL;
        link.textContent = "Cookie settings";
        links.appendChild(link);
    }

    window.MEYELensConsent = {
        get: getStoredConsent,
        set: setConsent
    };

    function removeBanner() {
        const banner = document.getElementById("cookie-consent-banner");
        if (banner) {
            banner.remove();
        }
    }

    function injectBanner() {
        if (document.getElementById("cookie-consent-banner")) {
            return;
        }

        const banner = document.createElement("div");
        banner.id = "cookie-consent-banner";
        banner.innerHTML = `
            <div class="cookie-consent-inner">
                <div class="cookie-consent-copy">
                    <strong>Cookies and analytics</strong>
                    <p>
                        We use Google Analytics only to understand website traffic and improve the site.
                        Analytics cookies will be activated only if you accept.
                        <a href="${PRIVACY_URL}">Privacy notice</a>
                    </p>
                </div>
                <div class="cookie-consent-actions">
                    <a class="btn" href="${SETTINGS_URL}" id="cookie-options">Options</a>
                    <button type="button" class="btn btn-primary" id="cookie-accept">Accept</button>
                </div>
            </div>
        `;

        const style = document.createElement("style");
        style.textContent = `
            #cookie-consent-banner {
                position: fixed;
                left: 16px;
                right: 16px;
                bottom: 16px;
                z-index: 9999;
                border: 1px solid var(--line, #44474a);
                border-radius: 12px;
                background: var(--bg-soft, #2c2e31);
                box-shadow: 0 10px 30px rgba(0,0,0,0.25);
            }

            .cookie-consent-inner {
                display: flex;
                align-items: flex-end;
                justify-content: space-between;
                gap: 18px;
                padding: 16px 18px;
            }

            .cookie-consent-copy strong {
                display: block;
                margin-bottom: 6px;
                color: var(--text, #d1d0c5);
                font-size: 14px;
            }

            .cookie-consent-copy p {
                margin: 0;
                color: var(--text-soft, #a3a099);
                font-size: 14px;
                line-height: 1.5;
                max-width: 62ch;
            }

            .cookie-consent-copy a {
                color: var(--accent, #e2b714);
                text-decoration: none;
            }

            .cookie-consent-copy a:hover {
                text-decoration: underline;
            }

            .cookie-consent-actions {
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
                flex: 0 0 auto;
            }

            @media (max-width: 760px) {
                .cookie-consent-inner {
                    flex-direction: column;
                    align-items: stretch;
                }

                .cookie-consent-actions {
                    width: 100%;
                }

                .cookie-consent-actions .btn {
                    flex: 1 1 auto;
                }
            }
        `;

        document.head.appendChild(style);
        document.body.appendChild(banner);

        document.getElementById("cookie-accept")?.addEventListener("click", () => setConsent(true));
    }

    function initConsent() {
        ensureDataLayer();
        injectSettingsLink();

        // Default denied until the user makes a choice
        window.gtag("consent", "default", {
            analytics_storage: "denied",
            ad_storage: "denied",
            ad_user_data: "denied",
            ad_personalization: "denied"
        });

        const saved = getStoredConsent();

        if (saved === "accepted") {
            enableAnalytics();
            return;
        }

        if (saved === "declined") {
            disableAnalytics();
            return;
        }

        if (!NO_BANNER_PAGES.includes(window.location.pathname.replace(/\.html$/, ""))) {
            injectBanner();
        }
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initConsent);
    } else {
        initConsent();
    }
})();