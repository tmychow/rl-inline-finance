export interface CompletionResponse {
  completion: string | null;
  tool_calls: ToolCall[];
  latency: number;
  model: string;
}

export interface ToolCall {
  tool: string;
  arguments: Record<string, any>;
  result: any;
  timestamp: string;
}

export interface TestCase {
  input: string;
  ground_truth: string;
}

export interface EvaluationResult {
  input: string;
  ground_truth: string;
  predictions: Record<string, {
    completion: string | null;
    is_correct: boolean;
    reasoning: string;
    latency: number;
    tool_calls: number;
    trace?: ToolCall[];
  }>;
}

export interface Model {
  provider: string;
  name: string;
  displayName: string;
}

export interface FinancialData {
  ticker: string;
  metric_name: string;
  period: string;
  value: number;
  unit: string;
  description?: string;
}

export interface Metric {
  metric_name: string;
  description: string;
  unit: string;
}