import axios from 'axios';

const api = axios.create({
    baseURL: '/api',
    headers: {
        'Content-Type': 'application/json',
    },
});

// Add request interceptor for auth token
api.interceptors.request.use((config) => {
    const token = localStorage.getItem('token');
    if (token) {
        config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
});

// Q&A API
export const askQuestion = async (question, conversationId = null) => {
    const response = await api.post('/qa/ask', {
        question,
        conversation_id: conversationId,
    });
    return response.data;
};

export const getConversations = async () => {
    const response = await api.get('/qa/conversations');
    return response.data;
};

export const getConversationHistory = async (conversationId) => {
    const response = await api.get(`/qa/conversations/${conversationId}`);
    return response.data;
};

export const deleteConversation = async (conversationId) => {
    const response = await api.delete(`/qa/conversations/${conversationId}`);
    return response.data;
};

// Documents API
export const uploadDocument = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await api.post('/documents/upload', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });
    return response.data;
};

export const getDocuments = async () => {
    const response = await api.get('/documents/');
    return response.data;
};

export const deleteDocument = async (documentId) => {
    const response = await api.delete(`/documents/${documentId}`);
    return response.data;
};

// Auth API (mock for now)
export const login = async (email, password) => {
    // Mock login - in production, call real endpoint
    return { token: 'mock-token', user: { email } };
};

export default api;
