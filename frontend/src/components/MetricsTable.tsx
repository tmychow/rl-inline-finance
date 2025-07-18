import React from 'react';
import styled from '@emotion/styled';

const Table = styled.table`
  width: 100%;
  border-collapse: collapse;
  margin-top: 20px;
`;

const Th = styled.th`
  background-color: #f5f5f5;
  padding: 12px;
  text-align: left;
  border-bottom: 2px solid #ddd;
`;

const Td = styled.td`
  padding: 12px;
  border-bottom: 1px solid #eee;
`;

const ModelBadge = styled.span`
  background-color: #e0e0e0;
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 12px;
`;

interface LatencyMetric {
  timestamp: Date;
  latency: number;
  model: string;
}

interface MetricsTableProps {
  metrics: LatencyMetric[];
}

export const MetricsTable: React.FC<MetricsTableProps> = ({ metrics }) => {
  const sortedMetrics = [...metrics].sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());

  return (
    <div>
      <h3>Request Latency Metrics</h3>
      <Table>
        <thead>
          <tr>
            <Th>Timestamp</Th>
            <Th>Model</Th>
            <Th>Latency (ms)</Th>
          </tr>
        </thead>
        <tbody>
          {sortedMetrics.map((metric, index) => (
            <tr key={index}>
              <Td>{metric.timestamp.toLocaleTimeString()}</Td>
              <Td>
                <ModelBadge>{metric.model}</ModelBadge>
              </Td>
              <Td>{(metric.latency * 1000).toFixed(0)} ms</Td>
            </tr>
          ))}
        </tbody>
      </Table>
    </div>
  );
};