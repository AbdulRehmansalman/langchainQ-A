import { useState, useEffect } from 'react';
import { Upload, File, Trash2, CheckCircle, XCircle, Loader } from 'lucide-react';
import { uploadDocument, getDocuments, deleteDocument } from '../api';
import './DocumentUpload.css';

function DocumentUpload() {
    const [documents, setDocuments] = useState([]);
    const [uploading, setUploading] = useState(false);
    const [dragActive, setDragActive] = useState(false);

    useEffect(() => {
        loadDocuments();
    }, []);

    const loadDocuments = async () => {
        try {
            const data = await getDocuments();
            setDocuments(data);
        } catch (error) {
            console.error('Failed to load documents:', error);
        }
    };

    const handleFileUpload = async (file) => {
        if (!file) return;

        setUploading(true);
        try {
            await uploadDocument(file);
            loadDocuments();
        } catch (error) {
            console.error('Upload failed:', error);
            alert('Upload failed. Please try again.');
        } finally {
            setUploading(false);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setDragActive(false);

        const file = e.dataTransfer.files[0];
        if (file) {
            handleFileUpload(file);
        }
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        setDragActive(true);
    };

    const handleDragLeave = () => {
        setDragActive(false);
    };

    const handleFileSelect = (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFileUpload(file);
        }
    };

    const handleDelete = async (documentId) => {
        if (!confirm('Are you sure you want to delete this document?')) return;

        try {
            await deleteDocument(documentId);
            loadDocuments();
        } catch (error) {
            console.error('Delete failed:', error);
            alert('Delete failed. Please try again.');
        }
    };

    const formatFileSize = (bytes) => {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    };

    const getStatusIcon = (status) => {
        switch (status) {
            case 'indexed':
                return <CheckCircle size={20} className="status-icon success" />;
            case 'processing':
                return <Loader size={20} className="status-icon processing spin" />;
            case 'failed':
                return <XCircle size={20} className="status-icon error" />;
            default:
                return null;
        }
    };

    return (
        <div className="document-upload">
            <div className="upload-header">
                <h2>Document Management</h2>
                <p className="text-secondary">
                    Upload documents to build your knowledge base
                </p>
            </div>

            {/* Upload Area */}
            <div
                className={`upload-area ${dragActive ? 'drag-active' : ''} ${uploading ? 'uploading' : ''}`}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
            >
                <Upload size={48} />
                <h3>Drop files here or click to browse</h3>
                <p>Supported formats: PDF, TXT, MD</p>
                <input
                    type="file"
                    accept=".pdf,.txt,.md"
                    onChange={handleFileSelect}
                    disabled={uploading}
                    style={{ display: 'none' }}
                    id="file-input"
                />
                <label htmlFor="file-input" className="upload-btn">
                    {uploading ? 'Uploading...' : 'Select File'}
                </label>
            </div>

            {/* Documents List */}
            <div className="documents-section">
                <h3>Uploaded Documents ({documents.length})</h3>

                {documents.length === 0 ? (
                    <div className="empty-state">
                        <File size={48} />
                        <p>No documents uploaded yet</p>
                    </div>
                ) : (
                    <div className="documents-grid">
                        {documents.map(doc => (
                            <div key={doc.id} className="document-card fade-in">
                                <div className="document-icon">
                                    <File size={32} />
                                </div>
                                <div className="document-info">
                                    <div className="document-name">{doc.filename}</div>
                                    <div className="document-meta">
                                        <span>{formatFileSize(doc.file_size)}</span>
                                        <span>•</span>
                                        <span>{doc.file_type.toUpperCase()}</span>
                                        {doc.chunk_count && (
                                            <>
                                                <span>•</span>
                                                <span>{doc.chunk_count} chunks</span>
                                            </>
                                        )}
                                    </div>
                                    <div className="document-status">
                                        {getStatusIcon(doc.status)}
                                        <span>{doc.status}</span>
                                    </div>
                                </div>
                                <button
                                    className="delete-doc-btn"
                                    onClick={() => handleDelete(doc.id)}
                                >
                                    <Trash2 size={18} />
                                </button>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* Info Section */}
            <div className="info-section">
                <h4>How it works</h4>
                <ol>
                    <li>Upload your documents (PDF, TXT, or MD files)</li>
                    <li>Documents are automatically split into chunks</li>
                    <li>Chunks are embedded and indexed in the vector store</li>
                    <li>Start asking questions in the Chat tab!</li>
                </ol>
            </div>
        </div>
    );
}

export default DocumentUpload;
