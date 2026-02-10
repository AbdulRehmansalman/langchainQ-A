import { useState } from 'react';
import { BarChart3, TrendingUp, Target, CheckCircle2 } from 'lucide-react';
import './EvaluationDashboard.css';

function EvaluationDashboard() {
    // Mock evaluation data - in production, fetch from backend
    const [metrics] = useState({
        retrieval: {
            recall_at_k: 0.85,
            precision_at_k: 0.92,
            mrr: 0.78,
            k: 5
        },
        generation: {
            faithfulness: 0.91,
            answer_relevance: 0.88,
            context_utilization: 0.84
        },
        overall_score: 0.87
    });

    const MetricCard = ({ title, value, description, icon: Icon, color }) => (
        <div className="metric-card fade-in">
            <div className="metric-header">
                <div className={`metric-icon ${color}`}>
                    <Icon size={24} />
                </div>
                <div className="metric-value">{(value * 100).toFixed(1)}%</div>
            </div>
            <div className="metric-title">{title}</div>
            <div className="metric-description">{description}</div>
            <div className="metric-bar">
                <div
                    className={`metric-fill ${color}`}
                    style={{ width: `${value * 100}%` }}
                />
            </div>
        </div>
    );

    return (
        <div className="evaluation-dashboard">
            <div className="dashboard-header">
                <h2>Evaluation Metrics</h2>
                <p className="text-secondary">
                    Comprehensive RAG system performance metrics
                </p>
            </div>

            {/* Overall Score */}
            <div className="overall-score-card">
                <div className="overall-content">
                    <div className="overall-icon">
                        <TrendingUp size={48} />
                    </div>
                    <div>
                        <div className="overall-label">Overall System Score</div>
                        <div className="overall-value">
                            {(metrics.overall_score * 100).toFixed(1)}%
                        </div>
                        <div className="overall-description">
                            Weighted average of all metrics
                        </div>
                    </div>
                </div>
            </div>

            {/* Retrieval Metrics */}
            <div className="metrics-section">
                <h3>
                    <Target size={20} />
                    Retrieval Metrics
                </h3>
                <p className="section-description">
                    How well does the system retrieve relevant documents?
                </p>
                <div className="metrics-grid">
                    <MetricCard
                        title={`Recall@${metrics.retrieval.k}`}
                        value={metrics.retrieval.recall_at_k}
                        description="% of relevant docs retrieved"
                        icon={CheckCircle2}
                        color="primary"
                    />
                    <MetricCard
                        title={`Precision@${metrics.retrieval.k}`}
                        value={metrics.retrieval.precision_at_k}
                        description="% of retrieved docs that are relevant"
                        icon={Target}
                        color="success"
                    />
                    <MetricCard
                        title="MRR"
                        value={metrics.retrieval.mrr}
                        description="Mean Reciprocal Rank - position of first relevant doc"
                        icon={TrendingUp}
                        color="warning"
                    />
                </div>
            </div>

            {/* Generation Metrics */}
            <div className="metrics-section">
                <h3>
                    <BarChart3 size={20} />
                    Generation Metrics
                </h3>
                <p className="section-description">
                    How good is the generated answer quality?
                </p>
                <div className="metrics-grid">
                    <MetricCard
                        title="Faithfulness"
                        value={metrics.generation.faithfulness}
                        description="Answer is grounded in context (no hallucinations)"
                        icon={CheckCircle2}
                        color="primary"
                    />
                    <MetricCard
                        title="Answer Relevance"
                        value={metrics.generation.answer_relevance}
                        description="Answer addresses the question"
                        icon={Target}
                        color="success"
                    />
                    <MetricCard
                        title="Context Utilization"
                        value={metrics.generation.context_utilization}
                        description="Effective use of retrieved context"
                        icon={TrendingUp}
                        color="warning"
                    />
                </div>
            </div>

            {/* Metrics Explanation */}
            <div className="metrics-explanation">
                <h3>Understanding the Metrics</h3>

                <div className="explanation-section">
                    <h4>Retrieval Metrics</h4>
                    <ul>
                        <li>
                            <strong>Recall@K:</strong> What percentage of all relevant documents were retrieved in the top K results?
                            <br />
                            <em>High recall = Found most relevant docs</em>
                        </li>
                        <li>
                            <strong>Precision@K:</strong> What percentage of the top K retrieved documents are actually relevant?
                            <br />
                            <em>High precision = Most retrieved docs are relevant</em>
                        </li>
                        <li>
                            <strong>MRR (Mean Reciprocal Rank):</strong> How quickly do we find the first relevant document?
                            <br />
                            <em>High MRR = Relevant docs appear at the top</em>
                        </li>
                    </ul>
                </div>

                <div className="explanation-section">
                    <h4>Generation Metrics</h4>
                    <ul>
                        <li>
                            <strong>Faithfulness:</strong> Is the answer grounded in the retrieved context?
                            <br />
                            <em>High faithfulness = No hallucinations, trustworthy answers</em>
                        </li>
                        <li>
                            <strong>Answer Relevance:</strong> Does the answer directly address the user's question?
                            <br />
                            <em>High relevance = On-topic, useful answers</em>
                        </li>
                        <li>
                            <strong>Context Utilization:</strong> How well does the answer use the retrieved context?
                            <br />
                            <em>High utilization = Comprehensive, detailed answers</em>
                        </li>
                    </ul>
                </div>

                <div className="explanation-section">
                    <h4>Why Evaluation Matters</h4>
                    <ul>
                        <li>📊 <strong>Measure Quality:</strong> Quantify system performance objectively</li>
                        <li>🔄 <strong>Compare Approaches:</strong> A/B test different RAG strategies</li>
                        <li>📈 <strong>Track Progress:</strong> Monitor improvements over time</li>
                        <li>🎯 <strong>Identify Issues:</strong> Find weak points in the system</li>
                        <li>✅ <strong>Build Trust:</strong> Demonstrate reliability to stakeholders</li>
                    </ul>
                </div>

                <div className="explanation-section">
                    <h4>Human Evaluation</h4>
                    <p>
                        While automated metrics are valuable, <strong>human evaluation is still crucial</strong>:
                    </p>
                    <ul>
                        <li>Captures nuance that automated metrics miss</li>
                        <li>Validates automated scores with real user perspective</li>
                        <li>Identifies edge cases and unusual scenarios</li>
                        <li>Provides qualitative feedback for improvement</li>
                    </ul>
                    <p className="recommendation">
                        <strong>Recommendation:</strong> Use automated metrics for continuous monitoring,
                        and validate with human evaluation on a sample of queries regularly.
                    </p>
                </div>
            </div>
        </div>
    );
}

export default EvaluationDashboard;
