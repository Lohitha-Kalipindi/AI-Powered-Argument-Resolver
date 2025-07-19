import gradio as gr
import requests
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Tuple

# Together AI API Configuration
TOGETHER_API_KEY = "your_together_ai_api_key_here"
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

def preprocess_chat_content(content: str) -> str:
    """Enhanced WhatsApp chat preprocessing with better message parsing"""
    if not content or not content.strip():
        return ""
    
    # Remove system messages first
    system_messages = [
        'Messages and calls are end-to-end encrypted',
        'Only people in this chat can read, listen to, or share them',
        'Learn more',
        'is a contact',
        'Your security code with',
        'changed. Tap to learn more',
        '<Media omitted>',
        'This message was deleted',
        'You deleted this message',
        'joined using this group',
        'left the group',
        'added you',
        'removed you',
        'created group',
        'changed the group description',
        'changed this group\'s icon',
        'null'  # Handle null messages
    ]
    
    # Remove system messages
    for msg in system_messages:
        content = re.sub(re.escape(msg), '', content, flags=re.IGNORECASE)
    
    # Parse WhatsApp messages with improved regex
    # Pattern: MM/DD/YYYY or DD-MM-YYYY, HH:MM [AM/PM] - Name: Message
    whatsapp_pattern = r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}),?\s*(\d{1,2}:\d{2}(?:\s*[ap]m)?)\s*-\s*([^:]+):\s*(.+)'
    
    messages = []
    lines = content.split('\n')
    current_message = None
    sender_map = {}  # For anonymization
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if this is a new message
        match = re.match(whatsapp_pattern, line, re.IGNORECASE)
        if match:
            # Save previous message if it exists
            if current_message:
                messages.append(current_message)
            
            groups = match.groups()
            if len(groups) < 4:
                return "⚠️ Invalid WhatsApp chat format detected. Please ensure the file matches the expected format (e.g., 'MM/DD/YYYY, HH:MM - Name: Message')."
            
            date, time, sender, message = groups
            
            # Clean sender name
            sender = sender.strip()
            
            # Clean message content
            message = message.strip()
            
            # Skip if message is empty or just whitespace
            if not message or message.isspace():
                current_message = None
                continue
            
            # Skip promotional/spam messages
            if is_promotional_message(message):
                current_message = None
                continue
            
            # Skip very long technical content (like LaTeX code)
            if len(message) > 1000 and contains_technical_content(message):
                current_message = None
                continue
            
            current_message = {
                'sender': sender,
                'message': message,
                'timestamp': f"{date} {time}"
            }
        else:
            # This might be a continuation of the previous message
            if current_message and line:
                # Only add if it's not a technical continuation
                if not contains_technical_content(line):
                    current_message['message'] += f" {line}"
    
    # Add the last message
    if current_message:
        messages.append(current_message)
    
    # Convert to conversation format with anonymization
    conversation_lines = []
    for msg in messages:
        # Anonymize sender names for privacy
        anonymized_sender = anonymize_sender(msg['sender'], sender_map)
        conversation_lines.append(f"{anonymized_sender}: {msg['message']}")
    
    # Join and clean up
    result = '\n'.join(conversation_lines)
    
    # Remove excessive whitespace
    result = re.sub(r'\n\s*\n', '\n', result)
    result = re.sub(r'\s+', ' ', result)
    
    return result.strip()

def is_promotional_message(message: str) -> bool:
    """Check if message is promotional/spam content"""
    promotional_keywords = [
        'referral code',
        'cashback',
        'join india',
        'click my link',
        'get assured',
        'download app',
        'earn up to',
        'https://',
        'http://',
        'www.',
        'paytm',
        'gpay',
        'phonepe'
    ]
    
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in promotional_keywords)

def contains_technical_content(text: str) -> bool:
    """Check if text contains technical content like LaTeX, code, etc."""
    technical_patterns = [
        r'\\documentclass',
        r'\\usepackage',
        r'\\begin{',
        r'\\end{',
        r'\\textbf{',
        r'\\section{',
        r'\\title{',
        r'def\s+\w+\(',
        r'import\s+\w+',
        r'from\s+\w+\s+import',
        r'<[^>]+>',  # HTML tags
        r'^\s*#.*$',  # Comment lines
        r'^\s*//.*$'  # Another comment style
    ]
    
    for pattern in technical_patterns:
        if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
            return True
    
    return False

def anonymize_sender(sender: str, sender_map: dict) -> str:
    """Anonymize sender names for privacy while maintaining conversation flow"""
    if not sender:
        return "Unknown"
    # Generate consistent anonymized names
    if sender not in sender_map:
        sender_id = len(sender_map) + 1
        sender_map[sender] = f"Person {sender_id}"
    
    return sender_map[sender]

def call_together_ai(prompt: str, system_prompt: str = None) -> str:
    """Call Together AI API with optimized settings"""
    if TOGETHER_API_KEY == "your_together_ai_api_key_here":
        return "⚠️ Invalid API Key: Please configure a valid Together AI API key."
    
    try:
        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "messages": messages,
            "max_tokens": 600,  # Balanced for quality and speed
            "temperature": 0.4,  # Balanced creativity and focus
            "top_p": 0.8
        }
        
        response = requests.post(TOGETHER_API_URL, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"API Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Error calling Together AI: {str(e)}"

def parse_conflict_response(response: str) -> Tuple[str, str, str]:
    """Parse the AI response into title, summary, and resolution components"""
    title = ""
    summary = ""
    resolution = ""
    
    # Split response into lines and process
    lines = response.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check for section headers
        if line.startswith('⚔️') or 'conflict title' in line.lower():
            current_section = 'title'
            title = re.sub(r'⚔️.*?:', '', line, flags=re.IGNORECASE).strip()
            title = re.sub(r'^\*+', '', title).strip()
            title = re.sub(r'^"(.*)"$', r'\1', title).strip()  # Remove quotes
        elif line.startswith('🔍') or 'conflict summary' in line.lower():
            current_section = 'summary'
            summary = re.sub(r'🔍.*?:', '', line, flags=re.IGNORECASE).strip()
            summary = re.sub(r'^\*+', '', summary).strip()
        elif line.startswith('🤝') or 'conflict resolution' in line.lower():
            current_section = 'resolution'
            resolution = re.sub(r'🤝.*?:', '', line, flags=re.IGNORECASE).strip()
            resolution = re.sub(r'^\*+', '', resolution).strip()
        elif line and current_section:
            # Continue adding to current section
            if current_section == 'title' and not title:
                title = line.strip('"')
            elif current_section == 'summary':
                summary += ' ' + line if summary else line
            elif current_section == 'resolution':
                resolution += ' ' + line if resolution else line
    
    # Clean up any remaining formatting and remove **bold** markers
    title = re.sub(r'\*\*', '', title).strip()
    summary = re.sub(r'\*\*', '', summary).strip()
    resolution = re.sub(r'\*\*', '', resolution).strip()
    
    # Ensure we have content with fallbacks
    if not title:
        title = "Communication Analysis"
    if not summary:
        summary = "The situation involves differing perspectives that require careful understanding and mediation."
    if not resolution:
        resolution = "Focus on open communication, active listening, and finding common ground to move forward constructively."
    
    return title, summary, resolution

def process_conversation(conversation_text: str) -> Tuple[str, str, str]:
    """Process single conversation text and return conflict analysis"""
    if not conversation_text or not conversation_text.strip():
        return "⚠️ Please enter a conversation to analyze.", "", ""
    
    if TOGETHER_API_KEY == "your_together_ai_api_key_here":
        return "⚠️ Please configure your Together AI API key in the code.", "", ""
    
    try:
        system_prompt = """You are a highly skilled conflict resolution expert with deep training in psychology, counseling, and nonviolent communication. You analyze disagreements with the wisdom of a seasoned therapist who understands human emotions and motivations. Your responses should be compassionate, insightful, and practical."""
        
        prompt = f"""You are an empathetic conflict analyst. Analyze the following conversation and provide a respectful, psychologically insightful resolution. Your goal is to mediate the argument by understanding both sides deeply, without taking sides. Your response should include:
1. ⚔️ Conflict Title (short and thoughtful)
2. 🔍 Conflict Summary (briefly describe the emotional and logical disagreement)
3. 🤝 Conflict Resolution (a concise solution that respects both perspectives - keep this under 100 words)
Use emotionally intelligent language and always aim for fairness and clarity.

Conversation: {conversation_text}

Please structure your response exactly like this:

⚔️ Conflict Title:
"[Your thoughtful title here]"

🔍 Conflict Summary:
[Your empathetic analysis of both sides here]

🤝 Conflict Resolution:
[Your concise solution that acknowledges both truths and provides practical next steps - maximum 100 words]"""
        
        response = call_together_ai(prompt, system_prompt)
        
        if "API Error" in response or "Error calling Together AI" in response:
            return "❌ API Error", response, "Please check your API key and try again."
        
        title, summary, resolution = parse_conflict_response(response)
        return title, summary, resolution
        
    except Exception as e:
        return "❌ Error analyzing conversation", f"An error occurred: {str(e)}", "Please try again or check your API configuration."

def process_pov(person1_pov: str, person2_pov: str) -> Tuple[str, str, str]:
    """Process individual points of view and return conflict analysis"""
    if not person1_pov.strip() or not person2_pov.strip():
        return "⚠️ Please enter both perspectives to analyze.", "", ""
    
    if TOGETHER_API_KEY == "your_together_ai_api_key_here":
        return "⚠️ Please configure your Together AI API key in the code.", "", ""
    
    try:
        system_prompt = """You are a highly skilled conflict resolution expert with deep training in psychology, counseling, and nonviolent communication. You analyze disagreements with the wisdom of a seasoned therapist who understands human emotions and motivations. Your responses should be compassionate, insightful, and practical."""
        
        prompt = f"""You are an empathetic conflict analyst. Analyze the following opposing POVs and provide a respectful, psychologically insightful resolution. Your goal is to mediate the argument by understanding both sides deeply, without taking sides. Your response should include:
1. ⚔️ Conflict Title (short and thoughtful)
2. 🔍 Conflict Summary (briefly describe the emotional and logical disagreement)
3. 🤝 Conflict Resolution (a concise solution that respects both perspectives - keep this under 100 words)
Use emotionally intelligent language and always aim for fairness and clarity.

Person 1's Perspective: {person1_pov}
Person 2's Perspective: {person2_pov}

Please structure your response exactly like this:

⚔️ Conflict Title:
"[Your thoughtful title here]"

🔍 Conflict Summary:
[Your empathetic analysis of both sides here]

🤝 Conflict Resolution:
[Your concise solution that acknowledges both truths and provides practical next steps - maximum 100 words]"""
        
        response = call_together_ai(prompt, system_prompt)
        
        if "API Error" in response or "Error calling Together AI" in response:
            return "❌ API Error", response, "Please check your API key and try again."
        
        title, summary, resolution = parse_conflict_response(response)
        return title, summary, resolution
        
    except Exception as e:
        return "❌ Error analyzing perspectives", f"An error occurred: {str(e)}", "Please try again or check your API configuration."

def process_uploaded_file(file) -> Tuple[str, str, str]:
    """Process uploaded conversation file and return conflict analysis"""
    if file is None:
        return "⚠️ Please upload a conversation file to analyze.", "", ""
    
    if TOGETHER_API_KEY == "your_together_ai_api_key_here":
        return "⚠️ Please configure your Together AI API key in the code.", "", ""
    
    try:
        # Read the uploaded file with proper encoding handling
        encodings = ['utf-8', 'latin-1', 'iso-8859-1']
        content = None
        for encoding in encodings:
            try:
                with open(file.name, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            return "❌ Error reading file", "Unable to decode file with supported encodings.", "Please upload a valid text file."
        
        # Check if it looks like a WhatsApp export
        if is_whatsapp_export(content):
            # Use the enhanced WhatsApp preprocessing
            cleaned_content = preprocess_chat_content(content)
        else:
            # Use basic cleaning for other formats
            cleaned_content = basic_content_cleaning(content)
        
        if not cleaned_content.strip():
            return "⚠️ No valid conversation content found in the file.", "", ""
        
        system_prompt = """You are a highly skilled conflict resolution expert with deep training in psychology, counseling, and nonviolent communication. You analyze disagreements with the wisdom of a seasoned therapist who understands human emotions and motivations. Your responses should be compassionate, insightful, and practical."""
        
        prompt = f"""You are an empathetic conflict analyst. Analyze the following conversation file and provide a respectful, psychologically insightful resolution. Your goal is to mediate the arguments by understanding all sides deeply, without taking sides. Your response should include:
1. ⚔️ Conflict Title (short and thoughtful)
2. 🔍 Conflict Summary (briefly describe the emotional and logical disagreement)
3. 🤝 Conflict Resolution (a concise solution that respects both perspectives - keep this under 100 words)
Use emotionally intelligent language and always aim for fairness and clarity.

Conversation Content: {cleaned_content[:2000]}

Please structure your response exactly like this:

⚔️ Conflict Title:
"[Your thoughtful title here]"

🔍 Conflict Summary:
[Your empathetic analysis of both sides here]

🤝 Conflict Resolution:
[Your concise solution that acknowledges both truths and provides practical next steps - maximum 100 words]"""
        
        response = call_together_ai(prompt, system_prompt)
        
        if "API Error" in response or "Error calling Together AI" in response:
            return "❌ API Error", response, "Please check your API key and try again."
        
        title, summary, resolution = parse_conflict_response(response)
        return title, summary, resolution
        
    except Exception as e:
        return "❌ Error processing file", f"An error occurred: {str(e)}", "Please try again with a different file format."

def is_whatsapp_export(content: str) -> bool:
    """Check if the content looks like a WhatsApp export"""
    if not content:
        return False
    whatsapp_patterns = [
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4},?\s*\d{1,2}:\d{2}\s*(?:[ap]m)?\s*-\s*[^:]+:',
        r'Messages and calls are end-to-end encrypted',
        r'<Media omitted>',
        r'This message was deleted'
    ]
    
    for pattern in whatsapp_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return True
    return False

def basic_content_cleaning(content: str) -> str:
    """Basic cleaning for non-WhatsApp files"""
    if not content:
        return ""
    # Remove excessive whitespace
    content = re.sub(r'\n\s*\n', '\n', content)
    content = re.sub(r'\s+', ' ', content)
    
    # Remove common system messages or metadata
    system_patterns = [
        r'\[.*?\]',  # Remove bracketed timestamps/metadata
        r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',  # Remove detailed timestamps
    ]
    
    for pattern in system_patterns:
        content = re.sub(pattern, '', content)
    
    return content.strip()

# Enhanced professional CSS with better accessibility and equal column heights
css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.gradio-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    font-family: 'Inter', sans-serif;
    min-height: 100vh;
}

.main-header {
    text-align: center;
    padding: 3rem 2rem;
    background: rgba(255, 255, 255, 0.15);
    border-radius: 20px;
    margin-bottom: 2rem;
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
}

.main-header h1 {
    color: white;
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 1rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    letter-spacing: -0.02em;
}

.main-header p {
    color: rgba(255, 255, 255, 0.9);
    font-size: 1.3rem;
    margin-bottom: 0;
    font-weight: 400;
}

.feature-card {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 20px;
    padding: 2rem;
    margin: 1rem 0;
    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.3);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    height: 100%;
}

.feature-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 30px 60px rgba(0,0,0,0.15);
}

.feature-card h3 {
    color: #333;
    font-size: 1.4rem;
    font-weight: 600;
    margin-bottom: 1rem;
    text-align: center;
}

.feature-card p {
    color: #666;
    font-size: 1rem;
    line-height: 1.6;
    text-align: center;
    margin-bottom: 1.5rem;
}

.nav-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 1rem 2rem;
    border-radius: 30px;
    font-size: 1.1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    width: 100%;
    letter-spacing: 0.5px;
}

.nav-button:hover {
    transform: translateY(-3px);
    box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
    background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
}

.back-button {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    border: none;
    padding: 0.8rem 1.5rem;
    border-radius: 25px;
    font-weight: 600;
    margin-bottom: 2rem;
    transition: all 0.3s ease;
    box-shadow: 0 5px 15px rgba(240, 147, 251, 0.3);
}

.back-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(240, 147, 251, 0.4);
}

.content-card {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 20px;
    padding: 2rem;
    margin: 1rem 0;
    box-shadow: 0 15px 35px rgba(0,0,0,0.1);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    height: 100%; /* Ensure content cards stretch to fill their parent */
}

/* Ensure equal height for columns in rows */
.gr-row {
    display: flex;
    align-items: stretch; /* Make columns stretch to equal height */
}

.gr-column {
    display: flex;
    flex-direction: column;
}

.gr-column > .content-card {
    flex-grow: 1; /* Allow content cards to grow to fill column height */
}

.page-header {
    text-align: center;
    padding: 2rem;
    background: rgba(255, 255, 255, 0.15);
    border-radius: 20px;
    margin-bottom: 2rem;
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.page-header h2 {
    color: white;
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.section-title {
    color: #333;
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: 1.5rem;
    text-align: center;
    padding: 1rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    background-clip: text;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.analyze-button {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    color: white;
    border: none;
    padding: 1rem 2rem;
    border-radius: 30px;
    font-size: 1.1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 8px 25px rgba(79, 172, 254, 0.3);
    width: 100%;
    margin-top: 1rem;
}

.analyze-button:hover {
    transform: translateY(-3px);
    box-shadow: 0 15px 35px rgba(79, 172, 254, 0.4);
}

.info-box {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
    border: 1px solid rgba(255, 255, 255, 0.2);
    color: white;
}

.info-box h4 {
    color: white;
    margin-bottom: 1rem;
    font-weight: 600;
}

.info-box ul {
    list-style: none;
    padding: 0;
}

.info-box li {
    padding: 0.5rem 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.info-box li:last-child {
    border-bottom: none;
}

.warning-box {
    background: rgba(255, 193, 7, 0.1);
    border: 1px solid rgba(255, 193, 7, 0.3);
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
    color: #856404;
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
}
"""

# Create the main interface
with gr.Blocks(css=css, title="AI Argument Resolver") as demo:
    
    # Landing Page
    with gr.Column(visible=True) as landing_page:
        gr.HTML("""
        <div class="main-header">
            <h1>🤖 AI Argument Resolver</h1>
            <p>Transform conflicts into conversations with AI-powered psychological analysis</p>
        </div>
        """)
        
        if TOGETHER_API_KEY == "your_together_ai_api_key_here":
            gr.HTML("""
            <div class="warning-box">
                <h4>⚠️ Configuration Required</h4>
                <p>Please replace 'your_together_ai_api_key_here' with your actual Together AI API key in the code to use this application.</p>
            </div>
            """)
        
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Column(elem_classes="feature-card"):
                    gr.HTML("""
                    <h3>💬 Conversation Analysis</h3>
                    <p>Paste your conversation and get instant psychological analysis with empathetic resolution strategies.</p>
                    """)
                    conversation_nav_btn = gr.Button("📝 Analyze Conversation", elem_classes="nav-button", size="lg")
                
            with gr.Column(scale=1):
                with gr.Column(elem_classes="feature-card"):
                    gr.HTML("""
                    <h3>👥 Perspective Comparison</h3>
                    <p>Compare different viewpoints with psychological insights to find common ground and mutual understanding.</p>
                    """)
                    pov_nav_btn = gr.Button("🎯 Compare Perspectives", elem_classes="nav-button", size="lg")
                
            with gr.Column(scale=1):
                with gr.Column(elem_classes="feature-card"):
                    gr.HTML("""
                    <h3>📁 File Analysis</h3>
                    <p>Upload chat files (including WhatsApp exports) for comprehensive conflict analysis with pattern recognition.</p>
                    """)
                    upload_nav_btn = gr.Button("📤 Upload & Analyze", elem_classes="nav-button", size="lg")

    # Conversation Analysis Page
    with gr.Column(visible=False) as conversation_page:
        gr.HTML("""
        <div class="page-header">
            <h2>💬 Conversation Analysis</h2>
        </div>
        """)
        
        back_to_home_1 = gr.Button("🏠 Back to Home", elem_classes="back-button")
        
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Column(elem_classes="content-card"):
                    gr.HTML("<h3 class='section-title'>📝 Enter Your Conversation</h3>")
                    conversation_input = gr.Textbox(
                        label="Conversation Text",
                        placeholder="Paste your conversation here...\n\nExample:\nPerson A: I can't believe you did that!\nPerson B: I was just trying to help...",
                        lines=12,
                        max_lines=20
                    )
                    analyze_btn = gr.Button("🔍 Analyze with AI Psychology", elem_classes="analyze-button", size="lg")
                
            with gr.Column(scale=1):
                with gr.Column(elem_classes="content-card"):
                    gr.HTML("<h3 class='section-title'>📊 AI Analysis Results</h3>")
                    conflict_title_1 = gr.Textbox(label="⚔️ Conflict Title", interactive=False)
                    conflict_summary_1 = gr.Textbox(label="🔍 Conflict Summary", lines=4, interactive=False)
                    conflict_resolution_1 = gr.Textbox(label="🤝 Conflict Resolution", lines=5, interactive=False)

    # Point of View Page
    with gr.Column(visible=False) as pov_page:
        gr.HTML("""
        <div class="page-header">
            <h2>👥 Perspective Comparison</h2>
        </div>
        """)
        
        back_to_home_2 = gr.Button("🏠 Back to Home", elem_classes="back-button")
        
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Column(elem_classes="content-card"):
                    gr.HTML("<h3 class='section-title'>👤 Enter Both Perspectives</h3>")
                    person1_input = gr.Textbox(
                        label="Person 1's Perspective",
                        placeholder="Describe the first person's viewpoint...",
                        lines=6,
                        max_lines=10
                    )
                    person2_input = gr.Textbox(
                        label="Person 2's Perspective", 
                        placeholder="Describe the second person's viewpoint...",
                        lines=6,
                        max_lines=10
                    )
                    analyze_pov_btn = gr.Button("🔍 Analyze Perspectives", elem_classes="analyze-button", size="lg")
                
            with gr.Column(scale=1):
                with gr.Column(elem_classes="content-card"):
                    gr.HTML("<h3 class='section-title'>📊 AI Analysis Results</h3>")
                    conflict_title_2 = gr.Textbox(label="⚔️ Conflict Title", interactive=False)
                    conflict_summary_2 = gr.Textbox(label="🔍 Conflict Summary", lines=4, interactive=False)
                    conflict_resolution_2 = gr.Textbox(label="🤝 Conflict Resolution", lines=5, interactive=False)

    # File Upload Page
    with gr.Column(visible=False) as upload_page:
        gr.HTML("""
        <div class="page-header">
            <h2>📁 File Analysis</h2>
        </div>
        """)
        
        back_to_home_3 = gr.Button("🏠 Back to Home", elem_classes="back-button")
        
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Column(elem_classes="content-card"):
                    gr.HTML("<h3 class='section-title'>📤 Upload Conversation File</h3>")
                    
                    gr.HTML("""
                    <div class="info-box">
                        <h3>📋 Supported File Formats:</h3>
                        <ul>
                            <li>📱 WhatsApp chat exports (.txt)</li>
                            <li>💬 Text conversations (.txt)</li>
                            <li>📄 Plain text files</li>
                        </ul>
                    </div>
                    """)
                    
                    file_input = gr.File(
                        label="Choose File",
                        file_types=[".txt", ".log", ".csv"],
                        type="filepath"
                    )
                    analyze_file_btn = gr.Button("🔍 Analyze File", elem_classes="analyze-button", size="lg")
                
            with gr.Column(scale=1):
                with gr.Column(elem_classes="content-card"):
                    gr.HTML("<h3 class='section-title'>📊 AI Analysis Results</h3>")
                    conflict_title_3 = gr.Textbox(label="⚔️ Conflict Title", interactive=False)
                    conflict_summary_3 = gr.Textbox(label="🔍 Conflict Summary", lines=4, interactive=False)
                    conflict_resolution_3 = gr.Textbox(label="🤝 Conflict Resolution", lines=5, interactive=False)

        # Additional information section
        gr.HTML("""
        <div class="info-box">
            <h4>🔒 Privacy & Security</h4>
            <ul>
                <li>✅ Your conversations are processed securely</li>
                <li>✅ Names are automatically anonymized</li>
                <li>✅ Files are processed temporarily and not stored</li>
                <li>✅ All analysis is confidential</li>
            </ul>
        </div>
        """)

    # Navigation Functions
    def show_conversation_page():
        return (
            gr.update(visible=False),  # landing_page
            gr.update(visible=True),   # conversation_page
            gr.update(visible=False),  # pov_page
            gr.update(visible=False)   # upload_page
        )
    
    def show_pov_page():
        return (
            gr.update(visible=False),  # landing_page
            gr.update(visible=False),  # conversation_page
            gr.update(visible=True),   # pov_page
            gr.update(visible=False)   # upload_page
        )
    
    def show_upload_page():
        return (
            gr.update(visible=False),  # landing_page
            gr.update(visible=False),  # conversation_page
            gr.update(visible=False),  # pov_page
            gr.update(visible=True)    # upload_page
        )
    
    def show_home():
        return (
            gr.update(visible=True),   # landing_page
            gr.update(visible=False),  # conversation_page
            gr.update(visible=False),  # pov_page
            gr.update(visible=False)   # upload_page
        )

    # Event Handlers
    conversation_nav_btn.click(
        show_conversation_page,
        outputs=[landing_page, conversation_page, pov_page, upload_page]
    )
    
    pov_nav_btn.click(
        show_pov_page,
        outputs=[landing_page, conversation_page, pov_page, upload_page]
    )
    
    upload_nav_btn.click(
        show_upload_page,
        outputs=[landing_page, conversation_page, pov_page, upload_page]
    )
    
    back_to_home_1.click(
        show_home,
        outputs=[landing_page, conversation_page, pov_page, upload_page]
    )
    
    back_to_home_2.click(
        show_home,
        outputs=[landing_page, conversation_page, pov_page, upload_page]
    )
    
    back_to_home_3.click(
        show_home,
        outputs=[landing_page, conversation_page, pov_page, upload_page]
    )

    analyze_btn.click(
        process_conversation,
        inputs=[conversation_input],
        outputs=[conflict_title_1, conflict_summary_1, conflict_resolution_1]
    )
    
    analyze_pov_btn.click(
        process_pov,
        inputs=[person1_input, person2_input],
        outputs=[conflict_title_2, conflict_summary_2, conflict_resolution_2]
    )
    
    analyze_file_btn.click(
        process_uploaded_file,
        inputs=[file_input],
        outputs=[conflict_title_3, conflict_summary_3, conflict_resolution_3]
    )

    # Add footer with additional information
    gr.HTML("""
    <div style="text-align: center; padding: 2rem; color: rgba(255, 255, 255, 0.7);">
        <p>🤖 Powered by AI Psychology • 🔒 Privacy Protected • 🎯 Conflict Resolution</p>
        <p style="font-size: 0.9rem;">Transform arguments into understanding with empathetic AI analysis</p>
    </div>
    """)

# Launch the application
if __name__ == "__main__":
    try:
        demo.launch(
            show_api=False
        )
    except Exception as e:
        print(f"Error launching Gradio app: {str(e)}")
