import React, { useState } from 'react';
import { FileText, Database, Brain, Server, Shield, BarChart3, MessageSquare, TrendingUp } from 'lucide-react';

const SystemArchitecture = () => {
  const [selectedComponent, setSelectedComponent] = useState(null);

  const components = [
    {
      id: 'data',
      title: 'Data Layer',
      icon: Database,
      color: 'bg-blue-500',
      details: [
        'E-commerce dataset ingestion',
        'Feature engineering pipeline',
        'Data validation & quality checks',
        'Train/test/validation splits'
      ]
    },
    {
      id: 'prediction',
      title: 'Predictive Models',
      icon: TrendingUp,
      color: 'bg-green-500',
      details: [
        'Discount prediction (XGBoost/LightGBM)',
        'Sales forecasting',
        'Customer conversion prediction',
        'Campaign performance models'
      ]
    },
    {
      id: 'llm',
      title: 'LLM Assistant',
      icon: Brain,
      color: 'bg-purple-500',
      details: [
        'Fine-tuned open-source LLM',
        'Domain adaptation (LoRA/QLoRA)',
        'Context-aware responses',
        'Multi-turn conversations'
      ]
    },
    {
      id: 'rag',
      title: 'RAG System',
      icon: FileText,
      color: 'bg-orange-500',
      details: [
        'Vector database (FAISS/Chroma)',
        'Semantic search over products',
        'Grounded response generation',
        'Source attribution'
      ]
    },
    {
      id: 'api',
      title: 'API Layer',
      icon: Server,
      color: 'bg-cyan-500',
      details: [
        'FastAPI endpoints',
        'Request validation',
        'Rate limiting',
        'Response caching'
      ]
    },
    {
      id: 'monitoring',
      title: 'Monitoring & Security',
      icon: Shield,
      color: 'bg-red-500',
      details: [
        'Model drift detection',
        'Performance tracking',
        'Input sanitization',
        'Explainability (SHAP)'
      ]
    }
  ];

  const keyFeatures = [
    { name: 'Discount Prediction', endpoint: '/predict_discount', method: 'POST' },
    { name: 'Q&A Assistant', endpoint: '/answer_question', method: 'POST' },
    { name: 'Product Search', endpoint: '/search_products', method: 'GET' },
    { name: 'Model Metrics', endpoint: '/metrics', method: 'GET' }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            Marketing Data Intelligence System
          </h1>
          <p className="text-xl text-gray-300">
            AI-Powered E-Commerce Analytics & Customer Assistant
          </p>
        </div>

        {/* Architecture Components */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
          {components.map((component) => {
            const Icon = component.icon;
            const isSelected = selectedComponent === component.id;
            
            return (
              <div
                key={component.id}
                onClick={() => setSelectedComponent(isSelected ? null : component.id)}
                className={`${component.color} rounded-xl p-6 cursor-pointer transition-all duration-300 transform hover:scale-105 ${
                  isSelected ? 'ring-4 ring-white shadow-2xl' : 'shadow-lg'
                }`}
              >
                <div className="flex items-center mb-4">
                  <Icon className="w-8 h-8 mr-3" />
                  <h3 className="text-xl font-bold">{component.title}</h3>
                </div>
                
                {isSelected && (
                  <ul className="space-y-2 mt-4 text-sm">
                    {component.details.map((detail, idx) => (
                      <li key={idx} className="flex items-start">
                        <span className="mr-2">•</span>
                        <span>{detail}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            );
          })}
        </div>

        {/* API Endpoints */}
        <div className="bg-slate-800 rounded-xl p-8 shadow-2xl mb-12">
          <div className="flex items-center mb-6">
            <MessageSquare className="w-8 h-8 mr-3 text-blue-400" />
            <h2 className="text-3xl font-bold">API Endpoints</h2>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {keyFeatures.map((feature, idx) => (
              <div key={idx} className="bg-slate-700 rounded-lg p-4 hover:bg-slate-600 transition-colors">
                <div className="flex justify-between items-center mb-2">
                  <h4 className="font-semibold text-lg">{feature.name}</h4>
                  <span className={`px-3 py-1 rounded text-xs font-bold ${
                    feature.method === 'POST' ? 'bg-green-600' : 'bg-blue-600'
                  }`}>
                    {feature.method}
                  </span>
                </div>
                <code className="text-sm text-gray-300">{feature.endpoint}</code>
              </div>
            ))}
          </div>
        </div>

        {/* Technology Stack */}
        <div className="bg-gradient-to-r from-purple-900 to-blue-900 rounded-xl p-8 shadow-2xl">
          <div className="flex items-center mb-6">
            <BarChart3 className="w-8 h-8 mr-3 text-purple-400" />
            <h2 className="text-3xl font-bold">Technology Stack</h2>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            {[
              { name: 'LLM', tech: 'Llama 3 / Mistral' },
              { name: 'ML', tech: 'XGBoost / LightGBM' },
              { name: 'Vector DB', tech: 'FAISS / ChromaDB' },
              { name: 'API', tech: 'FastAPI' },
              { name: 'Fine-tuning', tech: 'LoRA / QLoRA' },
              { name: 'Monitoring', tech: 'Prometheus / Grafana' },
              { name: 'Container', tech: 'Docker' },
              { name: 'Explainability', tech: 'SHAP' }
            ].map((item, idx) => (
              <div key={idx} className="bg-white bg-opacity-10 rounded-lg p-4 backdrop-blur">
                <div className="text-sm text-gray-300 mb-1">{item.name}</div>
                <div className="font-bold">{item.tech}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Implementation Steps */}
        <div className="mt-12 text-center text-gray-400">
          <p className="text-sm">
            Click on components above to see implementation details
          </p>
        </div>
      </div>
    </div>
  );
};

export default SystemArchitecture;