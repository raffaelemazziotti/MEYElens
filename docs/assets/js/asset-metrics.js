import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.119.0';

const supabase = createClient(
    'https://fqmatvvqllkmhtnxibgp.supabase.co',
    'sb_publishable_bFrnywKmVo_QuSbsbcal7w_W-9ddhZl'
);

const registeredLinks = new WeakSet();

export async function recordAssetMetric(assetKind, assetId, metric) {
    const { error } = await supabase.rpc('increment_asset_metric', {
        p_asset_kind: assetKind,
        p_asset_id: assetId,
        p_metric: metric
    });

    if (error) {
        console.warn('Asset metric was not recorded:', error.message);
    }
}

export async function loadAssetMetricCounts(assetKind, metric) {
    const { data, error } = await supabase
        .from('asset_metrics')
        .select('asset_id, count')
        .eq('asset_kind', assetKind)
        .eq('metric', metric);

    if (error) {
        console.warn('Asset metrics could not be loaded:', error.message);
        return new Map();
    }

    return new Map((data || []).map((item) => [item.asset_id, item.count]));
}

export function registerDownloadLink(link, assetKind, assetId) {
    if (!link || registeredLinks.has(link)) return;

    registeredLinks.add(link);

    link.addEventListener('click', (event) => {
        // Let modified clicks keep their normal browser behaviour. The metric is
        // still requested, but navigation must not be delayed for a new tab.
        if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) {
            void recordAssetMetric(assetKind, assetId, 'download');
            return;
        }

        event.preventDefault();

        // Wait briefly so the request is not cancelled by navigation. This counts
        // a requested download, not confirmation that a remote file completed.
        void Promise.race([
            recordAssetMetric(assetKind, assetId, 'download'),
            new Promise((resolve) => window.setTimeout(resolve, 600))
        ]).finally(() => {
            window.location.assign(link.href);
        });
    });
}

function registerDeclaredDownloadLinks() {
    document.querySelectorAll('[data-asset-kind][data-asset-id][data-asset-metric="download"]').forEach((link) => {
        registerDownloadLink(link, link.dataset.assetKind, link.dataset.assetId);
    });
}

async function showDeclaredDownloadCounts() {
    const links = [...document.querySelectorAll('[data-asset-kind][data-asset-id][data-asset-metric="download"]')];
    if (!links.length) return;

    const countsByKind = new Map();
    for (const kind of new Set(links.map((link) => link.dataset.assetKind))) {
        countsByKind.set(kind, await loadAssetMetricCounts(kind, 'download'));
    }

    links.forEach((link) => {
        const counts = countsByKind.get(link.dataset.assetKind);

        // An empty result means the counts could not be loaded, so show nothing.
        if (!counts || counts.size === 0) return;

        const downloads = Number(counts.get(link.dataset.assetId)) || 0;
        const wrapper = document.createElement('span');
        wrapper.className = 'download-with-count';

        const label = document.createElement('span');
        label.className = 'download-count';
        label.textContent = `${downloads} ${downloads === 1 ? 'download' : 'downloads'}`;

        link.replaceWith(wrapper);
        wrapper.append(link, label);
    });
}

function initDeclaredDownloadLinks() {
    registerDeclaredDownloadLinks();
    void showDeclaredDownloadCounts();
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initDeclaredDownloadLinks, { once: true });
} else {
    initDeclaredDownloadLinks();
}
