// js/main.js
const messagesDiv = document.getElementById('messages');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');

let conversationHistory = [];

function addMessage(content, role) {
  const msgDiv = document.createElement('div');
  msgDiv.className = `message ${role}`;
  
  if (role === 'typing') {
    msgDiv.className = 'message ai typing';
    msgDiv.innerHTML = '<span>Thinking...</span>';
  } else {
    msgDiv.textContent = content;
  }
  messagesDiv.appendChild(msgDiv);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

async function callLLM() {
  sendBtn.disabled = true;
  addMessage("", "typing");

  try {
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
  conversationHistory.push({ role: "user", content: text });  // ← THIS LINE FIXES EVERYTHING
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