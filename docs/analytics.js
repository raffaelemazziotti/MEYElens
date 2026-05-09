import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

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
  '/privacy.html',
  '/404.html'
])

async function recordPageView() {
  const path = window.location.pathname

  if (!allowedPaths.has(path)) {
    return
  }

  const { error } = await supabase
    .from('page_views')
    .insert([{ path }])

  if (error) {
    console.warn('Page view not recorded:', error.message)
  }
}

recordPageView()