import React from 'react';
import styled from '@emotion/styled';
import { ToolCall } from '../types';

const LogContainer = styled.div`
  max-height: 300px;
  overflow-y: auto;
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 12px;
  margin-top: 20px;
  background-color: #f9f9f9;
`;

const LogEntry = styled.div`
  margin-bottom: 12px;
  padding: 8px;
  background-color: white;
  border-radius: 4px;
  font-family: 'Monaco', 'Consolas', monospace;
  font-size: 14px;
`;

const ToolName = styled.span<{ isLLM?: boolean }>`
  font-weight: bold;
  color: ${props => props.isLLM ? '#9b59b6' : '#0066cc'};
`;

const Timestamp = styled.span`
  color: #666;
  font-size: 12px;
  float: right;
`;

const Arguments = styled.pre`
  margin: 4px 0;
  color: #333;
  font-size: 12px;
  overflow-x: auto;
`;

const Result = styled.pre`
  margin: 4px 0;
  color: #008000;
  font-size: 12px;
  overflow-x: auto;
`;

interface ActivityLogProps {
  toolCalls: ToolCall[];
}

export const ActivityLog: React.FC<ActivityLogProps> = ({ toolCalls }) => {
  return (
    <LogContainer>
      <h3>Activity Log</h3>
      {toolCalls.length === 0 ? (
        <p>No tool calls yet...</p>
      ) : (
        toolCalls.map((call, index) => {
          const isLLMDecision = call.tool === 'llm_decision';
          
          return (
            <LogEntry key={index}>
              <div>
                <ToolName isLLM={isLLMDecision}>
                  {isLLMDecision ? '🤖 LLM Response' : call.tool}
                </ToolName>
                <Timestamp>{new Date(call.timestamp).toLocaleTimeString()}</Timestamp>
              </div>
              {isLLMDecision && call.arguments.prompt && (
                <Arguments>Prompt: {call.arguments.prompt}</Arguments>
              )}
              {isLLMDecision && call.arguments.response && (
                <Result>Response: {call.arguments.response}</Result>
              )}
              {!isLLMDecision && Object.keys(call.arguments).length > 0 && (
                <Arguments>Args: {JSON.stringify(call.arguments, null, 2)}</Arguments>
              )}
              {!isLLMDecision && call.result !== undefined && (
                <Result>Result: {JSON.stringify(call.result, null, 2)}</Result>
              )}
            </LogEntry>
          );
        })
      )}
    </LogContainer>
  );
};