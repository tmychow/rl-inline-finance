import React, { useState, useRef, useEffect, KeyboardEvent } from 'react';
import styled from '@emotion/styled';
import { api } from '../api';
import { CompletionResponse, Model, ToolCall } from '../types';

const EditorContainer = styled.div`
  position: relative;
  width: 100%;
  max-width: 800px;
  margin: 0 auto;
`;

const EditorWrapper = styled.div`
  position: relative;
`;

const TextArea = styled.textarea`
  width: 100%;
  min-height: 200px;
  padding: 12px;
  font-size: 16px;
  font-family: 'Monaco', 'Consolas', monospace;
  border: 1px solid #ddd;
  border-radius: 4px;
  resize: vertical;
  background: transparent;
  z-index: 2;
  position: relative;
  color: #000;
  line-height: 1.5;
`;

const SuggestionOverlay = styled.div`
  position: absolute;
  top: 1px;
  left: 1px;
  right: 1px;
  padding: 12px;
  font-size: 16px;
  font-family: 'Monaco', 'Consolas', monospace;
  color: #999;
  pointer-events: none;
  white-space: pre-wrap;
  word-wrap: break-word;
  overflow-wrap: break-word;
  z-index: 1;
  line-height: 1.5;
  border: 1px solid transparent;
  border-radius: 4px;
`;

interface TextEditorProps {
  model: Model;
  onToolCall: (toolCall: ToolCall) => void;
  onLatencyUpdate: (latency: number, model: string) => void;
}

export const TextEditor: React.FC<TextEditorProps> = ({ model, onToolCall, onLatencyUpdate }) => {
  const [text, setText] = useState('');
  const [suggestion, setSuggestion] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const textAreaRef = useRef<HTMLTextAreaElement>(null);
  const debounceTimer = useRef<NodeJS.Timeout | null>(null);
  const lastProcessedText = useRef<string>('');
  
  // Use refs to avoid stale closures
  const modelRef = useRef(model);
  const onToolCallRef = useRef(onToolCall);
  const onLatencyUpdateRef = useRef(onLatencyUpdate);
  
  useEffect(() => {
    modelRef.current = model;
    onToolCallRef.current = onToolCall;
    onLatencyUpdateRef.current = onLatencyUpdate;
  }, [model, onToolCall, onLatencyUpdate]);

  useEffect(() => {
    if (debounceTimer.current) {
      clearTimeout(debounceTimer.current);
    }

    setSuggestion('');

    debounceTimer.current = setTimeout(async () => {
      if (text && !isLoading && text !== lastProcessedText.current) {
        lastProcessedText.current = text;
        setIsLoading(true);
        try {
          const response: CompletionResponse = await api.getCompletion(text, modelRef.current);
          
          console.log('API Response:', response); // Debug log
          
          if (response.completion) {
            setSuggestion(text + response.completion);
          } else {
            setSuggestion(''); // Clear suggestion if no completion
          }
          
          response.tool_calls.forEach(tc => onToolCallRef.current(tc));
          onLatencyUpdateRef.current(response.latency, response.model);
        } catch (error) {
          console.error('Error getting completion:', error);
        } finally {
          setIsLoading(false);
        }
      }
    }, 1000); // Increase debounce to 1 second

    return () => {
      if (debounceTimer.current) {
        clearTimeout(debounceTimer.current);
      }
    };
  }, [text]); // Only depend on text changes

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Tab' && suggestion && suggestion !== text) {
      e.preventDefault();
      setText(suggestion);
      setSuggestion('');
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value);
  };

  return (
    <EditorContainer>
      <EditorWrapper>
        <SuggestionOverlay>{suggestion}</SuggestionOverlay>
        <TextArea
          ref={textAreaRef}
          value={text}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          placeholder="Start typing financial queries... (e.g., 'Apple's revenue in')"
        />
      </EditorWrapper>
      {isLoading && <div style={{ marginTop: '8px', color: '#666' }}>Getting suggestions...</div>}
    </EditorContainer>
  );
};