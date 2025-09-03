import streamlit as st
import requests
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import time

class HealthcareChatbot:
    def __init__(self, api_base_url: str = "http://localhost:8000/api"):
        self.api_base_url = api_base_url
        self.chat_endpoint = f"{api_base_url}/chat"
        self.health_endpoint = f"{api_base_url}/chat/health"
        self.explain_endpoint = f"{api_base_url}/chat/explain-risk"
    
    def check_chatbot_health(self) -> Dict[str, Any]:
        """Check if chatbot service is available"""
        try:
            response = requests.get(self.health_endpoint, timeout=5)
            return response.json()
        except requests.exceptions.RequestException:
            return {"status": "error", "message": "Chatbot service unavailable"}
    
    def send_message(self, message: str, conversation_history: Optional[List[Dict]] = None, 
                    patient_context: Optional[Dict] = None, include_health_data: bool = False) -> Dict[str, Any]:
        """Send a message to the chatbot"""
        try:
            payload = {
                "message": message,
                "conversation_history": conversation_history or [],
                "patient_context": patient_context,
                "include_health_data": include_health_data
            }
            
            response = requests.post(
                self.chat_endpoint,
                json=payload,
                timeout=30,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error_detail = response.json().get("detail", "Unknown error")
                return {
                    "error": f"API Error ({response.status_code}): {error_detail}",
                    "response": "I'm sorry, I'm having trouble processing your request right now. Please try again later."
                }
                
        except requests.exceptions.Timeout:
            return {
                "error": "Request timeout",
                "response": "I'm taking longer than usual to respond. Please try again."
            }
        except requests.exceptions.RequestException as e:
            return {
                "error": f"Connection error: {str(e)}",
                "response": "I'm having trouble connecting to the AI service. Please check your connection and try again."
            }
        except Exception as e:
            return {
                "error": f"Unexpected error: {str(e)}",
                "response": "Something unexpected happened. Please try again."
            }
    
    def explain_risk_assessment(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get explanation for risk assessment results"""
        try:
            response = requests.post(
                self.explain_endpoint,
                json=risk_data,
                timeout=20,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Failed to get risk explanation: {response.status_code}"}
                
        except Exception as e:
            return {"error": f"Error explaining risk: {str(e)}"}

def process_message_automatically(message: str, include_health_data: bool = True, use_conversation_context: bool = True):
    """Helper function to automatically process a message through the AI"""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = HealthcareChatbot()
    
    # Add user message to history
    st.session_state.chat_messages.append({
        "role": "user",
        "content": message,
        "timestamp": datetime.now()
    })
    
    # Prepare conversation history for API
    conversation_history = []
    if use_conversation_context:
        for msg in st.session_state.chat_messages[-10:]:  # Last 10 messages
            if msg["role"] in ["user", "assistant"]:
                conversation_history.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
    
    # Get patient context if available
    patient_context = st.session_state.get('patient_data') if include_health_data else None
    
    try:
        # Send message to chatbot
        response_data = st.session_state.chatbot.send_message(
            message=message,
            conversation_history=conversation_history[:-1] if use_conversation_context else [],
            patient_context=patient_context,
            include_health_data=include_health_data
        )
        
        # Add assistant response to history
        assistant_msg = {
            "role": "assistant",
            "content": response_data.get("response", "I'm sorry, I couldn't process your request."),
            "timestamp": datetime.now(),
            "suggestions": response_data.get("suggestions", []),
            "requires_medical_attention": response_data.get("requires_medical_attention", False)
        }
        
        st.session_state.chat_messages.append(assistant_msg)
        
        # Show success toast
        if "error" not in response_data:
            st.toast("✅ AI response received!", icon="🤖")
        else:
            st.error(f"⚠️ {response_data['error']}")
    
    except Exception as e:
        st.error(f"Error processing message: {str(e)}")

def render_chatbot_interface():
    """Render the chatbot interface in Streamlit"""
    
    # Initialize chatbot in session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = HealthcareChatbot()
    
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    
    if 'chatbot_ready' not in st.session_state:
        st.session_state.chatbot_ready = False
    
    # Check chatbot health on first load
    if not st.session_state.chatbot_ready:
        health_status = st.session_state.chatbot.check_chatbot_health()
        st.session_state.chatbot_ready = health_status.get("status") == "healthy"
        st.session_state.chatbot_error = health_status.get("message", "") if not st.session_state.chatbot_ready else None
    
    st.markdown("## 🤖 Healthcare AI Assistant")
    
    # Show status
    if st.session_state.chatbot_ready:
        st.success("✅ AI Assistant is ready to help!")
    else:
        st.error(f"❌ AI Assistant unavailable: {st.session_state.get('chatbot_error', 'Unknown error')}")
        st.info("💡 **Setup Instructions:**\n1. Install required packages: `pip install openai`\n2. Set your OpenAI API key: `export OPENAI_API_KEY=your_key_here`\n3. Restart the backend server")
        return
    
    # Chat interface
    st.markdown("### 💬 Chat History")
    
    # Display conversation history
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_messages:
            st.markdown("""
            <div class='card' style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);'>
                <h4 style='color: #6c757d; margin-bottom: 1rem;'>👋 Start a conversation!</h4>
                <p style='color: #6c757d; margin: 0;'>Ask me anything about your health, risk assessment, or get personalized recommendations.</p>
            </div>
            """, unsafe_allow_html=True)
        
        for i, msg in enumerate(st.session_state.chat_messages):
            if msg["role"] == "user":
                st.markdown(f"""
                <div class='chat-message-user'>
                    <div class='chat-author chat-author-user'>
                        <span>👤</span> <strong>You</strong>
                    </div>
                    <div class='chat-content'>{msg['content']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Check if this is a medical attention response
                requires_attention = msg.get('requires_medical_attention', False)
                message_class = "medical-attention" if requires_attention else "normal"
                author_class = "medical-attention" if requires_attention else ""
                icon = "⚠️" if requires_attention else "🤖"
                
                st.markdown(f"""
                <div class='chat-message-assistant {message_class}'>
                    <div class='chat-author chat-author-assistant {author_class}'>
                        <span>{icon}</span> <strong>AI Assistant</strong>
                        {"<span style='margin-left: 0.5rem; font-size: 0.8rem; background: #ffcdd2; color: #c62828; padding: 0.2rem 0.5rem; border-radius: 12px;'>Medical Attention</span>" if requires_attention else ""}
                    </div>
                    <div class='chat-content'>{msg['content']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show suggestions if available
                if msg.get('suggestions'):
                    with st.expander("💡 Suggested follow-up questions", expanded=True):
                        st.markdown("<div class='chat-suggestions'>", unsafe_allow_html=True)
                        st.markdown("<p class='suggestions-title'>💬 Click any question below to ask it automatically:</p>", unsafe_allow_html=True)
                        
                        # Create columns for better layout
                        cols = st.columns(2) if len(msg['suggestions']) > 2 else [st.container()]
                        for idx, suggestion in enumerate(msg['suggestions']):
                            col_idx = idx % len(cols)
                            with cols[col_idx]:
                                suggestion_key = f"suggestion_{i}_{hash(suggestion)}"
                                if st.button(
                                    f"💭 {suggestion}", 
                                    key=suggestion_key, 
                                    use_container_width=True,
                                    help=f"Click to automatically ask: '{suggestion}'"
                                ):
                                    # Show processing indicator
                                    with st.spinner("🤔 Processing your question..."):
                                        # Automatically process the suggestion through AI
                                        process_message_automatically(suggestion, include_health_data=True, use_conversation_context=True)
                                    st.rerun()
                        st.markdown("</div>", unsafe_allow_html=True)
    
    # Input area with improved styling
    st.markdown("### 📝 Send a Message")
    st.markdown("<div class='chat-input-container'>", unsafe_allow_html=True)
    
    with st.form("chat_form", clear_on_submit=True):
        # Message input
        user_input = st.text_area(
            "Ask me anything about your health...", 
            placeholder="e.g., What does my risk assessment mean? How can I improve my heart health? What do these symptoms suggest?",
            key="chat_input",
            height=100,
            help="Type your health-related question here. I can help explain medical terms, analyze your health data, and provide general health guidance."
        )
        
        # Options row
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            include_health_data = st.checkbox(
                "📈 Include my health data", 
                value=True, 
                help="Allow the AI to reference your current health metrics for personalized responses"
            )
        
        with col2:
            conversation_context = st.checkbox(
                "📜 Use conversation history", 
                value=True,
                help="Allow the AI to reference previous messages in this conversation"
            )
            
        with col3:
            submit_button = st.form_submit_button(
                "💬 Send", 
                use_container_width=True,
                type="primary"
            )
        
        if submit_button and user_input.strip():
            # Add user message to history
            st.session_state.chat_messages.append({
                "role": "user",
                "content": user_input.strip(),
                "timestamp": datetime.now()
            })
            
            # Prepare conversation history for API
            conversation_history = []
            if conversation_context:
                for msg in st.session_state.chat_messages[-10:]:  # Last 10 messages
                    if msg["role"] in ["user", "assistant"]:
                        conversation_history.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
            
            # Get patient context if available
            patient_context = st.session_state.get('patient_data') if include_health_data else None
            
            # Show enhanced thinking indicator
            thinking_placeholder = st.empty()
            with thinking_placeholder:
                st.markdown("""
                <div class='thinking-indicator'>
                    <span>🤔 AI is analyzing your question</span>
                    <div class='thinking-dots'>
                        <div class='thinking-dot'></div>
                        <div class='thinking-dot'></div>
                        <div class='thinking-dot'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Send message to chatbot
                response_data = st.session_state.chatbot.send_message(
                    message=user_input.strip(),
                    conversation_history=conversation_history[:-1] if conversation_context else [],
                    patient_context=patient_context,
                    include_health_data=include_health_data
                )
            
            thinking_placeholder.empty()
            
            # Add assistant response to history
            assistant_msg = {
                "role": "assistant",
                "content": response_data.get("response", "I'm sorry, I couldn't process your request."),
                "timestamp": datetime.now(),
                "suggestions": response_data.get("suggestions", []),
                "requires_medical_attention": response_data.get("requires_medical_attention", False)
            }
            
            st.session_state.chat_messages.append(assistant_msg)
            
            # Show error if any
            if "error" in response_data:
                st.error(f"⚠️ {response_data['error']}")
            
            st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Quick action buttons
    if st.session_state.chat_messages:
        st.markdown("### Quick Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Clear Chat", help="Clear conversation history"):
                st.session_state.chat_messages = []
                st.rerun()
        
        with col2:
            if st.button("📊 Explain My Risk", help="Get detailed explanation of your risk assessment"):
                if 'last_prediction' in st.session_state:
                    # Create a detailed risk explanation request
                    risk_data = st.session_state.last_prediction
                    risk_message = f"Please explain my cardiovascular risk assessment. My risk level is {risk_data.get('risk_level', 'Unknown')} with a probability of {risk_data.get('risk_probability', 0):.1%}. What does this mean and what should I do?"
                    # Automatically process the risk explanation request
                    process_message_automatically(risk_message, include_health_data=True, use_conversation_context=True)
                    st.rerun()
                else:
                    st.warning("No risk assessment available. Please run a prediction first.")
        
        with col3:
            if st.button("💡 Health Tips", help="Get personalized health tips"):
                tip_message = "Can you give me some personalized health tips based on my current health data?"
                # Automatically process the health tips request
                process_message_automatically(tip_message, include_health_data=True, use_conversation_context=True)
                st.rerun()

def render_chatbot_sidebar():
    """Render a compact chatbot in the sidebar"""
    st.sidebar.markdown("## 🤖 Quick Health Chat")
    
    # Initialize if needed
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = HealthcareChatbot()
    
    # Quick health check
    health_status = st.session_state.chatbot.check_chatbot_health()
    if health_status.get("status") != "healthy":
        st.sidebar.error("AI Assistant offline")
        return
    
    # Simple chat input
    with st.sidebar.form("sidebar_chat"):
        quick_question = st.text_input("Ask a quick question:", placeholder="e.g., Is my blood pressure normal?")
        if st.form_submit_button("Ask"):
            if quick_question.strip():
                # Process the question through the main chatbot and add to chat history
                process_message_automatically(quick_question.strip(), include_health_data=True, use_conversation_context=True)
                st.sidebar.success("✅ Question added to main chat!")
                st.sidebar.info("📈 Check the AI Assistant page for the full conversation.")