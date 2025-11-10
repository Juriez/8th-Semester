// js/main.js
const messagesDiv = document.getElementById('messages');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const themeToggle = document.getElementById('themeToggle');

// Theme Management
function initTheme() {
  const savedTheme = localStorage.getItem('theme') || 'light';
  document.documentElement.setAttribute('data-theme', savedTheme);
}

function toggleTheme() {
  const currentTheme = document.documentElement.getAttribute('data-theme');
  const newTheme = currentTheme === 'light' ? 'dark' : 'light';
  document.documentElement.setAttribute('data-theme', newTheme);
  localStorage.setItem('theme', newTheme);
}

// Initialize theme on load
initTheme();

// Theme toggle event listener
themeToggle.addEventListener('click', toggleTheme);

// Seed the conversation with a lightweight system prompt for better answers
let conversationHistory = [
  { role: "system", content: "You are AI, a helpful and concise assistant. Use brief paragraphs and markdown when useful." }
];

function addMessage(content, role) {
  const msgDiv = document.createElement('div');
  msgDiv.className = `message ${role}`;
  
  if (role === 'typing') {
    msgDiv.className = 'message ai typing';
    msgDiv.innerHTML = '<span class="typing-dots"></span>';
  } else {
    // Enhanced markdown rendering for perfect readability
    let formatted = content
      // Headers (### Title)
      .replace(/###\s+(.+?)(?:\n|$)/g, '<strong style="display: block; font-size: 1.1em; margin: 12px 0 8px 0; color: var(--text-primary);">$1</strong>')
      // Bold (**text**)
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      // Italic (*text*)
      .replace(/\*(.+?)\*/g, '<em>$1</em>')
      // Code blocks (```code```)
      .replace(/```([\s\S]+?)```/g, '<code style="display: block; background: var(--input-bg); padding: 12px; border-radius: 8px; margin: 8px 0; font-family: monospace; font-size: 0.9em; overflow-x: auto;">$1</code>')
      // Inline code (`code`)
      .replace(/`([^`]+)`/g, '<code style="background: var(--input-bg); padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 0.9em;">$1</code>')
      // Bullet points (- item or * item)
      .replace(/^[\-\*]\s+(.+)$/gm, '<div style="margin-left: 16px; padding-left: 8px; border-left: 2px solid var(--accent); margin: 6px 0;">• $1</div>')
      // Numbered lists (1. item)
      .replace(/^\d+\.\s+(.+)$/gm, '<div style="margin-left: 16px; padding-left: 8px; margin: 6px 0;">$1</div>')
      // Line breaks
      .replace(/\n\n/g, '<br><br>')
      .replace(/\n/g, '<br>');
    
    msgDiv.innerHTML = formatted;
  }
  
  messagesDiv.appendChild(msgDiv);
  
  // Smooth scroll to bottom
  setTimeout(() => {
    messagesDiv.scrollTo({
      top: messagesDiv.scrollHeight,
      behavior: 'smooth'
    });
  }, 50);
}

async function callLLM() {
  sendBtn.disabled = true;
  addMessage("", "typing");

  try {
    if (!API_CONFIG || !API_CONFIG.url || !API_CONFIG.key || !API_CONFIG.model) {
      throw new Error("API configuration missing. Check config/api.js");
    }

    const response = await fetch(API_CONFIG.url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${API_CONFIG.key}`
      },
      body: JSON.stringify({
        model: API_CONFIG.model,
        messages: conversationHistory,  // Now always has at least 1 message
        temperature: 0.7,
        max_tokens: 1024
      })
    });

    if (!response.ok) {
      const err = await response.text();
      console.error("Groq Error:", response.status, err);
      document.querySelector('.message.typing')?.remove();
      addMessage(`Groq Error ${response.status}: ${err}`, "ai");
      sendBtn.disabled = false;
      return;
    }

    const data = await response.json();
    document.querySelector('.message.typing')?.remove();

    if (data.choices?.[0]?.message?.content) {
      const reply = data.choices[0].message.content.trim();
      addMessage(reply, "ai");
      conversationHistory.push({ role: "assistant", content: reply });
    } else {
      addMessage("Empty response. Check console.", "ai");
    }
  } catch (err) {
    document.querySelector('.message.typing')?.remove();
    addMessage("Network error: " + err.message, "ai");
    console.error(err);
  }
  sendBtn.disabled = false;
  userInput.focus();
}

function sendMessage() {
  const text = userInput.value.trim();
  if (!text) return;

  addMessage(text, "user");
  conversationHistory.push({ role: "user", content: text });  // store message in history
  userInput.value = "";
  callLLM();
}

userInput.addEventListener('keypress', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
sendBtn.addEventListener('click', sendMessage);

window.onload = () => userInput.focus();