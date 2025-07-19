import React, { useState } from 'react';
import styled from '@emotion/styled';
import { TextEditor } from './components/TextEditor';
import { ActivityLog } from './components/ActivityLog';
import { MetricsTable } from './components/MetricsTable';
import { Evaluation } from './components/Evaluation';
import { FundamentalsViewer } from './components/FundamentalsViewer';
import { Model, ToolCall } from './types';

const AppContainer = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
`;

const Header = styled.h1`
  text-align: center;
  color: #333;
`;

const TabContainer = styled.div`
  display: flex;
  border-bottom: 2px solid #ddd;
  margin-bottom: 20px;
`;

const Tab = styled.button<{ active: boolean }>`
  padding: 10px 20px;
  border: none;
  background: ${props => props.active ? '#0066cc' : 'transparent'};
  color: ${props => props.active ? 'white' : '#333'};
  cursor: pointer;
  font-size: 16px;
  
  &:hover {
    background: ${props => props.active ? '#0066cc' : '#f0f0f0'};
  }
`;

const ModelSelector = styled.div`
  margin: 20px 0;
  display: flex;
  align-items: center;
  gap: 10px;
`;

const Select = styled.select`
  padding: 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
`;

const ContentGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  margin-top: 20px;
`;

const models: Model[] = [
  { provider: 'openai', name: 'gpt-4.1', displayName: 'GPT-4.1' },
  { provider: 'ollama', name: 'gemma3:4b', displayName: 'Gemma 3 4B (Local)' },
];

type TabType = 'editor' | 'evaluation' | 'fundamentals';

function App() {
  const [activeTab, setActiveTab] = useState<TabType>('editor');
  const [selectedModel, setSelectedModel] = useState<Model>(models[0]);
  const [toolCalls, setToolCalls] = useState<ToolCall[]>([]);
  const [latencyMetrics, setLatencyMetrics] = useState<Array<{
    timestamp: Date;
    latency: number;
    model: string;
  }>>([]);

  const handleToolCall = (toolCall: ToolCall) => {
    setToolCalls(prev => [...prev, toolCall]);
  };

  const handleLatencyUpdate = (latency: number, model: string) => {
    setLatencyMetrics(prev => [...prev, {
      timestamp: new Date(),
      latency,
      model
    }]);
  };

  return (
    <AppContainer>
      <Header>AI Financial Assistant Demo</Header>
      
      <TabContainer>
        <Tab active={activeTab === 'editor'} onClick={() => setActiveTab('editor')}>
          Text Editor
        </Tab>
        <Tab active={activeTab === 'evaluation'} onClick={() => setActiveTab('evaluation')}>
          Evaluation
        </Tab>
        <Tab active={activeTab === 'fundamentals'} onClick={() => setActiveTab('fundamentals')}>
          Fundamentals Viewer
        </Tab>
      </TabContainer>

      {activeTab === 'editor' && (
        <>
          <ModelSelector>
            <label>Model:</label>
            <Select
              value={`${selectedModel.provider}/${selectedModel.name}`}
              onChange={(e) => {
                const [provider, name] = e.target.value.split('/');
                const model = models.find(m => m.provider === provider && m.name === name);
                if (model) setSelectedModel(model);
              }}
            >
              {models.map(model => (
                <option key={`${model.provider}/${model.name}`} value={`${model.provider}/${model.name}`}>
                  {model.displayName}
                </option>
              ))}
            </Select>
          </ModelSelector>

          <TextEditor
            model={selectedModel}
            onToolCall={handleToolCall}
            onLatencyUpdate={handleLatencyUpdate}
          />

          <ContentGrid>
            <ActivityLog toolCalls={toolCalls} />
            <MetricsTable metrics={latencyMetrics} />
          </ContentGrid>
        </>
      )}

      {activeTab === 'evaluation' && (
        <Evaluation models={models} />
      )}

      {activeTab === 'fundamentals' && (
        <FundamentalsViewer />
      )}
    </AppContainer>
  );
}

export default App;
