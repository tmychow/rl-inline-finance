import React, { useState, useEffect } from 'react';
import styled from '@emotion/styled';
import { api } from '../api';
import { FinancialData } from '../types';

const Container = styled.div`
  margin-top: 40px;
`;

const Select = styled.select`
  padding: 8px;
  margin: 10px;
  border: 1px solid #ddd;
  border-radius: 4px;
`;

const Table = styled.table`
  width: 100%;
  border-collapse: collapse;
  margin-top: 20px;
`;

const Th = styled.th`
  background-color: #f5f5f5;
  padding: 12px;
  text-align: left;
  border: 1px solid #ddd;
  position: sticky;
  top: 0;
  z-index: 10;
`;

const Td = styled.td`
  padding: 12px;
  border: 1px solid #ddd;
`;

const TableContainer = styled.div`
  max-height: 600px;
  overflow-y: auto;
  border: 1px solid #ddd;
  border-radius: 4px;
`;

export const FundamentalsViewer: React.FC = () => {
  const [tickers, setTickers] = useState<string[]>([]);
  const [selectedTicker, setSelectedTicker] = useState<string>('');
  const [financialData, setFinancialData] = useState<FinancialData[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadTickers();
  }, []);

  useEffect(() => {
    if (selectedTicker) {
      loadFinancialData(selectedTicker);
    }
  }, [selectedTicker]);

  const loadTickers = async () => {
    try {
      const response = await api.getTickers();
      setTickers(response.tickers);
      if (response.tickers.length > 0) {
        setSelectedTicker(response.tickers[0]);
      }
    } catch (error) {
      console.error('Error loading tickers:', error);
    }
  };

  const loadFinancialData = async (ticker: string) => {
    setLoading(true);
    try {
      const response = await api.getFinancialData(ticker);
      if ('data' in response) {
        setFinancialData(response.data);
      }
    } catch (error) {
      console.error('Error loading financial data:', error);
    } finally {
      setLoading(false);
    }
  };

  const groupedData = financialData.reduce((acc, item) => {
    if (!acc[item.period]) {
      acc[item.period] = {};
    }
    acc[item.period][item.metric_name] = {
      value: item.value,
      unit: item.unit,
      description: item.description
    };
    return acc;
  }, {} as Record<string, Record<string, any>>);

  const periods = Object.keys(groupedData).sort().reverse();
  const metrics = Array.from(new Set(financialData.map(d => d.metric_name))).sort();

  return (
    <Container>
      <h2>Fundamentals Viewer</h2>
      
      <div>
        <label>Select Ticker: </label>
        <Select 
          value={selectedTicker} 
          onChange={(e) => setSelectedTicker(e.target.value)}
        >
          {tickers.map(ticker => (
            <option key={ticker} value={ticker}>{ticker}</option>
          ))}
        </Select>
      </div>

      {loading ? (
        <p>Loading data...</p>
      ) : (
        <TableContainer>
          <Table>
            <thead>
              <tr>
                <Th>Metric</Th>
                {periods.map(period => (
                  <Th key={period}>{period}</Th>
                ))}
              </tr>
            </thead>
            <tbody>
              {metrics.map(metric => (
                <tr key={metric}>
                  <Td style={{ fontWeight: 'bold' }}>{metric}</Td>
                  {periods.map(period => {
                    const data = groupedData[period]?.[metric];
                    return (
                      <Td key={period}>
                        {data ? (
                          <>
                            {data.value.toLocaleString()}
                            {data.unit && ` ${data.unit}`}
                          </>
                        ) : '-'}
                      </Td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </Table>
        </TableContainer>
      )}
    </Container>
  );
};