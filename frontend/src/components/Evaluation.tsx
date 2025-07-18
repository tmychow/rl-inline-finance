import React, { useState } from 'react';
import styled from '@emotion/styled';
import { api } from '../api';
import { TestCase, Model, EvaluationResult } from '../types';

const Container = styled.div`
  margin-top: 40px;
`;

const FileInput = styled.input`
  margin: 10px 0;
`;

const Button = styled.button`
  background-color: #0066cc;
  color: white;
  padding: 10px 20px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  margin: 0 10px;

  &:hover {
    background-color: #0052a3;
  }

  &:disabled {
    background-color: #ccc;
    cursor: not-allowed;
  }
`;

const ResultsTable = styled.table`
  width: 100%;
  border-collapse: collapse;
  margin-top: 20px;
`;

const Th = styled.th`
  background-color: #f5f5f5;
  padding: 12px;
  text-align: left;
  border: 1px solid #ddd;
`;

const Td = styled.td`
  padding: 12px;
  border: 1px solid #ddd;
  vertical-align: top;
`;

const Check = styled.span`
  color: green;
  font-size: 20px;
`;

const Cross = styled.span`
  color: red;
  font-size: 20px;
`;

const AccuracyBox = styled.div`
  margin: 20px 0;
  padding: 15px;
  background-color: #f0f0f0;
  border-radius: 4px;
`;

interface EvaluationProps {
  models: Model[];
}

export const Evaluation: React.FC<EvaluationProps> = ({ models }) => {
  const [testCases, setTestCases] = useState<TestCase[]>([]);
  const [results, setResults] = useState<EvaluationResult[]>([]);
  const [accuracyScores, setAccuracyScores] = useState<Record<string, number>>({});
  const [isEvaluating, setIsEvaluating] = useState(false);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      try {
        const response = await api.uploadEvaluationCsv(file);
        setTestCases(response.test_cases);
      } catch (error) {
        console.error('Error uploading file:', error);
      }
    }
  };

  const runEvaluation = async () => {
    setIsEvaluating(true);
    try {
      const response = await api.batchEvaluation(testCases, models);
      setResults(response.results);
      setAccuracyScores(response.accuracy_scores);
    } catch (error) {
      console.error('Error running evaluation:', error);
    } finally {
      setIsEvaluating(false);
    }
  };

  return (
    <Container>
      <h2>Model Evaluation</h2>
      
      <div>
        <label>Upload CSV file (with 'input' and 'ground_truth' columns):</label>
        <FileInput type="file" accept=".csv" onChange={handleFileUpload} />
      </div>

      {testCases.length > 0 && (
        <div>
          <p>Loaded {testCases.length} test cases</p>
          <Button onClick={runEvaluation} disabled={isEvaluating}>
            {isEvaluating ? 'Running Evaluation...' : 'Run Evaluation'}
          </Button>
        </div>
      )}

      {Object.keys(accuracyScores).length > 0 && (
        <AccuracyBox>
          <h3>Overall Accuracy Scores</h3>
          {Object.entries(accuracyScores).map(([model, score]) => (
            <div key={model}>
              <strong>{model}:</strong> {(score * 100).toFixed(1)}%
            </div>
          ))}
        </AccuracyBox>
      )}

      {results.length > 0 && (
        <ResultsTable>
          <thead>
            <tr>
              <Th>Input</Th>
              <Th>Ground Truth</Th>
              {models.map(model => (
                <Th key={`${model.provider}/${model.name}`}>
                  {model.displayName}
                </Th>
              ))}
            </tr>
          </thead>
          <tbody>
            {results.map((result, idx) => (
              <tr key={idx}>
                <Td>{result.input}</Td>
                <Td>{result.ground_truth}</Td>
                {models.map(model => {
                  const modelKey = `${model.provider}/${model.name}`;
                  const prediction = result.predictions[modelKey];
                  return (
                    <Td key={modelKey}>
                      {prediction ? (
                        <>
                          <div>{prediction.completion || 'No completion'}</div>
                          <div>
                            {prediction.is_correct ? <Check>✓</Check> : <Cross>✗</Cross>}
                          </div>
                          <div style={{ fontSize: '12px', color: '#666' }}>
                            {(prediction.latency * 1000).toFixed(0)}ms
                          </div>
                        </>
                      ) : (
                        'N/A'
                      )}
                    </Td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </ResultsTable>
      )}
    </Container>
  );
};