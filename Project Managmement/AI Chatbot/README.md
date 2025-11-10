# AI Chatbot

A simple AI chatbot powered by Groq LLM API, reengineered by Mahir, Rafid & Mehedi.

## Features
- Clean and modern chat interface with **Light/Dark theme toggle**
- Powered by Groq's fast LLM models
- Conversation history maintained
- Real-time responses
- Theme preference saved in browser
- Beautiful color schemes for both themes

## Prerequisites
- A modern web browser (Chrome, Firefox, Safari, Edge)
- Python 3.x (for running local server)
- Groq API key (free tier available)

## Setup Instructions

### 1. Get Your Groq API Key
1. Visit [Groq Console](https://console.groq.com/keys)
2. Sign up or log in to your account
3. Create a new API key
4. Copy your API key

### 2. Configure the API Key
1. Open `config/api.js` in your favorite text editor
2. Replace the placeholder API key with your actual key:
```javascript
const API_CONFIG = {
  url: "https://api.groq.com/openai/v1/chat/completions",
  key: "YOUR_ACTUAL_API_KEY_HERE",  // Replace this!
  model: "llama-3.1-70b-versatile"
};
```

### 3. Run the Project

#### Quick Start (Recommended - Handles Port Conflicts!)
```bash
# Navigate to project directory
cd /home/bs00793/Desktop/SPM/SPM_VibeCoder

# Run the start script (automatically frees port 5500 if occupied)
./start.sh
```

The script will:
- ✅ Automatically kill any process using port 5500
- ✅ Start the server on `http://127.0.0.1:5500`
- ✅ Display instructions to open in browser

Then open your browser and go to: `http://127.0.0.1:5500/index.html`

---

#### Alternative Options:

**Option A: Using Python Directly**
```bash
# If port 5500 is occupied, first kill it:
lsof -ti:5500 | xargs kill -9

# Then run the server:
python3 -m http.server 5500
```

**Option B: Using a Different Port**
```bash
# Use any available port (e.g., 8000)
python3 -m http.server 8000

# Then open: http://127.0.0.1:8000/index.html
```

**Option C: Using Node.js (if installed)**
```bash
npm install -g http-server
http-server -p 5500
```

**Option D: Using VS Code Live Server Extension**
1. Install "Live Server" extension in VS Code
2. Right-click on `index.html`
3. Select "Open with Live Server"

## Usage
1. Type your message in the input box at the bottom
2. Click "Send" or press Enter
3. Wait for the AI to respond
4. Continue the conversation!

## Troubleshooting

### Port Already in Use (Error: Address already in use)
If you get `OSError: [Errno 98] Address already in use`:

**Solution 1: Use the start.sh script** (handles this automatically)
```bash
./start.sh
```

**Solution 2: Manually free the port**
```bash
# Find and kill process using port 5500
lsof -ti:5500 | xargs kill -9

# Then start the server
python3 -m http.server 5500
```

**Solution 3: Use a different port**
```bash
# Use port 8000 instead
python3 -m http.server 8000
# Open: http://127.0.0.1:8000/index.html
```

### Error 405: Method Not Allowed
- Make sure you're using a proper HTTP server (not just opening the HTML file directly)
- The browser's CORS policy requires a server

### Groq Error 401: Unauthorized
- Your API key is invalid or expired
- Get a new key from [Groq Console](https://console.groq.com/keys)

### Groq Error 429: Rate Limit
- You've exceeded the free tier limits
- Wait a few minutes and try again
- Consider upgrading your Groq plan

### Network Error
- Check your internet connection
- Verify the API endpoint is correct
- Check browser console for detailed errors

## Project Structure
```
SPM_VibeCoder/
├── index.html          # Main HTML file
├── config/
│   └── api.js         # API configuration (API key here!)
├── css/
│   └── style.css      # Styling
├── js/
│   └── main.js        # Application logic
└── README.md          # This file
```

## Security Note
⚠️ **Never commit your API key to version control!**
- Add `config/api.js` to `.gitignore`
- Use environment variables in production
- Regenerate your key if accidentally exposed

## Credits
Reengineered by: Mahir, Rafid & Mehedi  
Powered by: Groq LLM API

## License
Educational project for SPM course
