import { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { MessageSquare, Upload, BarChart3, FileText } from 'lucide-react';
import ChatInterface from './components/ChatInterface';
import DocumentUpload from './components/DocumentUpload';
import EvaluationDashboard from './components/EvaluationDashboard';
import './App.css';

function App() {
    const [activeTab, setActiveTab] = useState('chat');

    return (
        <Router>
            <div className="app">
                {/* Header */}
                <header className="app-header">
                    <div className="container">
                        <div className="header-content">
                            <div className="logo">
                                <MessageSquare size={32} />
                                <h1>Intelligent Q&A System</h1>
                            </div>
                            <div className="header-info">
                                <span className="badge">Advanced RAG</span>
                                <span className="badge">LangChain</span>
                            </div>
                        </div>
                    </div>
                </header>

                {/* Navigation */}
                <nav className="app-nav">
                    <div className="container">
                        <div className="nav-tabs">
                            <button
                                className={`nav-tab ${activeTab === 'chat' ? 'active' : ''}`}
                                onClick={() => setActiveTab('chat')}
                            >
                                <MessageSquare size={20} />
                                <span>Chat</span>
                            </button>
                            <button
                                className={`nav-tab ${activeTab === 'documents' ? 'active' : ''}`}
                                onClick={() => setActiveTab('documents')}
                            >
                                <Upload size={20} />
                                <span>Documents</span>
                            </button>
                            <button
                                className={`nav-tab ${activeTab === 'evaluation' ? 'active' : ''}`}
                                onClick={() => setActiveTab('evaluation')}
                            >
                                <BarChart3 size={20} />
                                <span>Evaluation</span>
                            </button>
                        </div>
                    </div>
                </nav>

                {/* Main Content */}
                <main className="app-main">
                    <div className="container">
                        {activeTab === 'chat' && <ChatInterface />}
                        {activeTab === 'documents' && <DocumentUpload />}
                        {activeTab === 'evaluation' && <EvaluationDashboard />}
                    </div>
                </main>

                {/* Footer */}
                <footer className="app-footer">
                    <div className="container">
                        <p>Built with LangChain • FastAPI • React</p>
                        <p className="text-secondary">
                            Demonstrating Advanced RAG Patterns
                        </p>
                    </div>
                </footer>
            </div>
        </Router>
    );
}

export default App;
