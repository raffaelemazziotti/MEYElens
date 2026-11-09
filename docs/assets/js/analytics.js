import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.119.0'

const supabaseUrl = 'https://fqmatvvqllkmhtnxibgp.supabase.co'
const supabaseKey = 'sb_publishable_bFrnywKmVo_QuSbsbcal7w_W-9ddhZl'
const supabase = createClient(supabaseUrl, supabaseKey)

const allowedPaths = new Set([
  '/',
  '/index.html',
  '/get_started.html',
  '/hardware.html',
  '/software.html',
  '/models.html',
  '/resources.html',
  '/contact.html',
  '/model-detail.html',
  '/tutorial-detail.html',
  '/privacy.html',
  '/404.html'
])

// Visits from a local test server or a file:// page are not real visitors.
// Run localStorage.setItem('meyelens_count_local', '1') in the console to
// count them anyway while testing the counters.
const isLocal =
  window.location.protocol === 'file:' ||
  ['localhost', '127.0.0.1', '[::1]'].includes(window.location.hostname)

async function recordPageView() {
  if (isLocal && localStorage.getItem('meyelens_count_local') !== '1') {
    return
  }

  const path = window.location.pathname === '/index.html' ? '/' : window.location.pathname

  if (!allowedPaths.has(path)) {
    return
  }

  const { error } = await supabase.rpc('increment_page_view', {
    page_path: path
  })

  if (error) {
    console.warn('Page view not recorded:', error.message)
  }
}

recordPageView()