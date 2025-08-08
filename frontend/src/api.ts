import axios from 'axios';
import { CompletionResponse, TestCase, Model, FinancialData, Metric } from './types';

const API_BASE_URL = 'http://localhost:8070/api';

export const api = {
  async getCompletion(text: string, model: Model): Promise<CompletionResponse> {
    const response = await axios.post(`${API_BASE_URL}/completion`, {
      text,
      model_provider: model.provider,
      model_name: model.name,
    });
    return response.data;
  },

  async batchEvaluation(testCases: TestCase[], models: Model[]) {
    const response = await axios.post(`${API_BASE_URL}/batch-evaluation`, {
      test_cases: testCases,
      models: models.map(m => ({ provider: m.provider, name: m.name })),
    });
    return response.data;
  },

  async uploadEvaluationCsv(file: File) {
    const formData = new FormData();
    formData.append('file', file);
    const response = await axios.post(`${API_BASE_URL}/upload-evaluation-csv`, formData);
    return response.data;
  },

  async getFinancialData(ticker?: string): Promise<{ data: FinancialData[] } | { tickers: any[] }> {
    const params = ticker ? { ticker } : {};
    const response = await axios.get(`${API_BASE_URL}/financial-data`, { params });
    return response.data;
  },

  async getTickers(): Promise<{ tickers: string[] }> {
    const response = await axios.get(`${API_BASE_URL}/tickers`);
    return response.data;
  },

  async getMetrics(): Promise<{ metrics: Metric[] }> {
    const response = await axios.get(`${API_BASE_URL}/metrics`);
    return response.data;
  },
};