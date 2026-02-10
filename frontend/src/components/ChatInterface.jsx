import { useState, useEffect, useRef } from 'react';
import { Send, Loader, MessageSquare, Trash2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { askQuestion, getConversations, getConversationHistory, deleteConversation } from '../api';
import './ChatInterface.css';

function ChatInterface() {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [conversations, setConversations] = useState([]);
    const [currentConversationId, setCurrentConversationId] = useState(null);
    const messagesEndRef = useRef(null);

    useEffect(() => {
        loadConversations();
    }, []);

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    const loadConversations = async () => {
        try {
            const data = await getConversations();
            setConversations(data);
        } catch (error) {
            console.error('Failed to load conversations:', error);
        }
    };

    const loadConversation = async (conversationId) => {
        try {
            const history = await getConversationHistory(conversationId);
            const formattedMessages = history.map(msg => ({
                role: msg.role,
                content: msg.content,
                sources: msg.sources
            }));
            setMessages(formattedMessages);
            setCurrentConversationId(conversationId);
        } catch (error) {
            console.error('Failed to load conversation:', error);
        }
    };

    const handleDeleteConversation = async (conversationId) => {
        try {
            await deleteConversation(conversationId);
            loadConversations();
            if (currentConversationId === conversationId) {
                setMessages([]);
                setCurrentConversationId(null);
            }
        } catch (error) {
            console.error('Failed to delete conversation:', error);
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!input.trim() || loading) return;

        const userMessage = { role: 'user', content: input };
        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setLoading(true);

        try {
            const response = await askQuestion(input, currentConversationId);

            const assistantMessage = {
                role: 'assistant',
                content: response.answer,
                sources: response.sources || []
            };

            setMessages(prev => [...prev, assistantMessage]);
            setCurrentConversationId(response.conversation_id);
            loadConversations();
        } catch (error) {
            const errorMessage = {
                role: 'assistant',
                content: 'Sorry, I encountered an error. Please try again.',
                error: true
            };
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setLoading(false);
        }
    };

    const startNewChat = () => {
        setMessages([]);
        setCurrentConversationId(null);
    };

    return (
        <div className="chat-interface">
            {/* Sidebar */}
            <div className="chat-sidebar">
                <button className="new-chat-btn" onClick={startNewChat}>
                    <MessageSquare size={20} />
                    <span>New Chat</span>
                </button>

                <div className="conversations-list">
                    <h3>Recent Conversations</h3>
                    {conversations.map(conv => (
                        <div
                            key={conv.id}
                            className={`conversation-item ${currentConversationId === conv.id ? 'active' : ''}`}
                        >
                            <div
                                className="conversation-content"
                                onClick={() => loadConversation(conv.id)}
                            >
                                <div className="conversation-title">{conv.title}</div>
                                <div className="conversation-meta">
                                    {conv.message_count} messages
                                </div>
                            </div>
                            <button
                                className="delete-btn"
                                onClick={(e) => {
                                    e.stopPropagation();
                                    handleDeleteConversation(conv.id);
                                }}
                            >
                                <Trash2 size={16} />
                            </button>
                        </div>
                    ))}
                </div>
            </div>

            {/* Main Chat Area */}
            <div className="chat-main">
                <div className="messages-container">
                    {messages.length === 0 ? (
                        <div className="empty-state">
                            <MessageSquare size={64} />
                            <h2>Start a Conversation</h2>
                            <p>Ask me anything about your documents!</p>
                            <div className="example-questions">
                                <button onClick={() => setInput("What is the refund policy?")}>
                                    What is the refund policy?
                                </button>
                                <button onClick={() => setInput("How do I contact support?")}>
                                    How do I contact support?
                                </button>
                            </div>
                        </div>
                    ) : (
                        messages.map((message, index) => (
                            <div
                                key={index}
                                className={`message ${message.role} ${message.error ? 'error' : ''} fade-in`}
                            >
                                <div className="message-content">
                                    <ReactMarkdown>{message.content}</ReactMarkdown>
                                    {message.sources && message.sources.length > 0 && (
                                        <div className="sources">
                                            <strong>Sources:</strong>
                                            {message.sources.map((source, idx) => (
                                                <span key={idx} className="source-tag">
                                                    {source.source}
                                                    {source.page && ` (p.${source.page})`}
                                                </span>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            </div>
                        ))
                    )}
                    {loading && (
                        <div className="message assistant fade-in">
                            <div className="message-content">
                                <div className="typing-indicator">
                                    <span></span>
                                    <span></span>
                                    <span></span>
                                </div>
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                {/* Input Form */}
                <form className="chat-input-form" onSubmit={handleSubmit}>
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Ask a question..."
                        disabled={loading}
                        className="chat-input"
                    />
                    <button
                        type="submit"
                        disabled={loading || !input.trim()}
                        className="send-btn"
                    >
                        {loading ? <Loader className="spin" size={20} /> : <Send size={20} />}
                    </button>
                </form>
            </div>
        </div>
    );
}

export default ChatInterface;
