import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const supabaseUrl = 'https://fqmatvvqllkmhtnxibgp.supabase.co'
const supabaseKey = 'sb_publishable_bFrnywKmVo_QuSbsbcal7w_W-9ddhZl'
const supabase = createClient(supabaseUrl, supabaseKey)

const form = document.getElementById('contactForm')
const statusEl = document.getElementById('contact-status')

function setStatus(message, type = '') {
    statusEl.textContent = message
    statusEl.className = 'form-status'
    if (type) statusEl.classList.add(type)
}

form?.addEventListener('submit', async (event) => {
    event.preventDefault()
    setStatus('Sending...')

    const payload = {
        name: document.getElementById('contact-name').value.trim() || null,
        email: document.getElementById('contact-email').value.trim(),
        message: document.getElementById('contact-message').value.trim()
    }

    if (!payload.email) {
        setStatus('Please enter your email.', 'error')
        return
    }

    if (!payload.message) {
        setStatus('Please enter a message.', 'error')
        return
    }

    const { error } = await supabase
        .from('contact_messages')
        .insert([payload])

    if (error) {
        setStatus(`Message not sent: ${error.message}`, 'error')
        return
    }

    form.reset()
    setStatus('Thank you - your message has been sent.', 'success')
})