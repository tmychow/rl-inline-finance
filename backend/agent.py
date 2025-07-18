import json
import re
import time
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import asyncio
from enum import Enum
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from backend.database import (
        get_all_tickers, get_all_metrics, get_financial_value,
        get_available_periods, get_latest_period
    )
except ImportError:
    from database import (
        get_all_tickers, get_all_metrics, get_financial_value,
        get_available_periods, get_latest_period
    )

class ToolName(Enum):
    GET_METRICS = "get_metrics"
    GET_TICKERS = "get_tickers"
    GET_VALUE = "get_value"
    CALCULATE = "calculate"
    RETURN_ANSWER = "return_answer"
    LLM_DECISION = "llm_decision"

class ToolCall:
    def __init__(self, tool_name: str, arguments: Dict[str, Any], result: Any = None):
        self.tool_name = tool_name
        self.arguments = arguments
        self.result = result
        self.timestamp = datetime.now().isoformat()

class FinancialAgent:
    def __init__(self):
        self.tool_calls: List[ToolCall] = []
        
    async def get_metrics(self) -> List[Dict[str, str]]:
        metrics = await get_all_metrics()
        result = [{"name": m["metric_name"], "description": m["description"], "unit": m["unit"]} for m in metrics]
        self.tool_calls.append(ToolCall(ToolName.GET_METRICS.value, {}, result))
        return result
    
    async def get_tickers(self) -> List[str]:
        tickers = await get_all_tickers()
        self.tool_calls.append(ToolCall(ToolName.GET_TICKERS.value, {}, tickers))
        return tickers
    
    async def get_value(self, metric: str, ticker: str, period: str) -> Optional[Dict[str, Any]]:
        # Handle "latest" period request
        if period.lower() == "latest":
            actual_period = await get_latest_period(ticker, metric)
            if actual_period:
                period = actual_period
            else:
                self.tool_calls.append(ToolCall(
                    ToolName.GET_VALUE.value, 
                    {"metric": metric, "ticker": ticker, "period": "latest"},
                    None
                ))
                return None
        
        value_data = await get_financial_value(ticker, metric, period)
        
        if value_data:
            # Add the period to the result for clarity
            value_data["period"] = period
        
        self.tool_calls.append(ToolCall(
            ToolName.GET_VALUE.value, 
            {"metric": metric, "ticker": ticker, "period": period},
            value_data
        ))
        return value_data
    
    def calculate(self, num1: float, num2: float, operation: str, duration: Optional[int] = None) -> float:
        result = None
        
        if operation == "add":
            result = num1 + num2
        elif operation == "subtract":
            result = num1 - num2
        elif operation == "multiply":
            result = num1 * num2
        elif operation == "divide":
            result = num1 / num2 if num2 != 0 else None
        elif operation == "CAGR" and duration:
            if num2 > 0 and num1 > 0 and duration > 0:
                result = ((num1 / num2) ** (1 / duration) - 1) * 100
        
        self.tool_calls.append(ToolCall(
            ToolName.CALCULATE.value,
            {"num1": num1, "num2": num2, "operation": operation, "duration": duration},
            result
        ))
        return result
    
    def return_answer(self, answer: str) -> str:
        self.tool_calls.append(ToolCall(
            ToolName.RETURN_ANSWER.value,
            {"answer": answer},
            answer
        ))
        return answer
    
    def record_llm_response(self, prompt: str, response: str, decision: str = None):
        """Record LLM responses for transparency"""
        self.tool_calls.append(ToolCall(
            ToolName.LLM_DECISION.value,
            {"prompt": prompt, "response": response},
            decision or response
        ))
    
    def clear_tool_calls(self):
        self.tool_calls = []
    
    def get_tool_calls_log(self) -> List[Dict[str, Any]]:
        return [
            {
                "tool": call.tool_name,
                "arguments": call.arguments,
                "result": call.result,
                "timestamp": call.timestamp
            }
            for call in self.tool_calls
        ]

class AgentExecutor:
    def __init__(self, model_provider: str = "openai", model_name: str = "gpt-4o-mini"):
        self.model_provider = model_provider
        self.model_name = model_name
        self.agent = FinancialAgent()
    
    async def _parse_tool_calls_from_response(self, response: str) -> List[Dict[str, Any]]:
        tool_calls = []
        
        tool_patterns = {
            "get_metrics": r"get_metrics\(\)",
            "get_tickers": r"get_tickers\(\)",
            "get_value": r"get_value\(['\"\"\"]([^'\"\"\"]+)['\"\"\"],\s*['\"\"\"]([^'\"\"\"]+)['\"\"\"],\s*['\"\"\"]([^'\"\"\"]+)['\"\"\"]\)",
            "calculate": r"calculate\(([^,]+),\s*([^,]+),\s*['\"\"\"]([^'\"\"\"]+)['\"\"\"](?:,\s*(\d+))?\)",
            "return_answer": r"return_answer\([\"\"\"\'](.*?)[\"\"\"\']\)"
        }
        
        for line in response.split('\n'):
            for tool_name, pattern in tool_patterns.items():
                match = re.search(pattern, line)
                if match:
                    if tool_name == "get_metrics" or tool_name == "get_tickers":
                        tool_calls.append({"tool": tool_name, "args": {}})
                    elif tool_name == "get_value":
                        tool_calls.append({
                            "tool": tool_name,
                            "args": {
                                "metric": match.group(1),
                                "ticker": match.group(2),
                                "period": match.group(3)
                            }
                        })
                    elif tool_name == "calculate":
                        tool_calls.append({
                            "tool": tool_name,
                            "args": {
                                "num1": float(match.group(1)),
                                "num2": float(match.group(2)),
                                "operation": match.group(3),
                                "duration": int(match.group(4)) if match.group(4) else None
                            }
                        })
                    elif tool_name == "return_answer":
                        tool_calls.append({
                            "tool": tool_name,
                            "args": {"answer": match.group(1)}
                        })
        
        return tool_calls
    
    async def _execute_tool_call(self, tool_call: Dict[str, Any]) -> Any:
        tool_name = tool_call["tool"]
        args = tool_call.get("args", {})
        
        if tool_name == "get_metrics":
            return await self.agent.get_metrics()
        elif tool_name == "get_tickers":
            return await self.agent.get_tickers()
        elif tool_name == "get_value":
            return await self.agent.get_value(**args)
        elif tool_name == "calculate":
            return self.agent.calculate(**args)
        elif tool_name == "return_answer":
            return self.agent.return_answer(**args)
        
        return None
    
    async def _call_model(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": prompt}
        ]
        return await self._call_model_with_messages(messages)
    
    async def _call_model_with_messages(self, messages: List[Dict[str, str]]) -> str:
        if self.model_provider == "openai":
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
            
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=500
            )
            return response.choices[0].message.content
            
        elif self.model_provider == "ollama":
            import ollama
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={"temperature": 0.1}
            )
            return response['message']['content']
        
        return ""
    
    def _get_system_prompt(self) -> str:
        return '''You are a financial data assistant that completes partial text with accurate financial data.

Available tools:
- get_metrics(): returns list of available metrics
- get_tickers(): returns list of available tickers  
- get_value(metric, ticker, period): gets a specific value
  - Period format: "2024Q1" for Q1 2024, "2024FY" for fiscal year 2024, or "latest" for most recent
- calculate(num1, num2, operation, duration): operations are "add", "subtract", "multiply", "divide", "CAGR"
- return_answer(answer): returns the final completion text (ONLY the completion, not the full text)

IMPORTANT: This is a multi-turn conversation. After you call tools, you'll receive the results and can call more tools or return the answer.

Example flow:
User: Complete this text with financial data: 'Apple's revenue in'
Assistant: get_value("revenue", "AAPL", "latest")
User: Tool results:
get_value: {"value": 383.3, "unit": "USD_billions", "period": "2024FY"}
Assistant: return_answer(" fiscal 2024 was $383.3 billion")

Another example:
User: Complete this text with financial data: 'Microsoft's Q2 2024 earnings per share was'
Assistant: get_value("eps", "MSFT", "2024Q2")
User: Tool results:
get_value: {"value": 2.93, "unit": "USD", "period": "2024Q2"}
Assistant: return_answer("$2.93")

Always use exact tool syntax. Only return the completion text in return_answer, not the full sentence.
If the text doesn't need financial data, return "NO_COMPLETION_NEEDED".'''
    
    async def get_completion(self, text: str) -> Tuple[Optional[str], List[Dict[str, Any]], float]:
        start_time = time.time()
        self.agent.clear_tool_calls()
        
        # Skip empty text
        if not text.strip():
            return None, [], time.time() - start_time
        
        try:
            # Build initial conversation
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": f"Complete this text with financial data: '{text}'"}
            ]
            
            # Multi-turn conversation loop
            max_turns = 10
            for turn in range(max_turns):
                # Get model response
                response = await self._call_model_with_messages(messages)
                print(f"Turn {turn + 1} - Model response: {response}")  # Debug
                
                # Record the LLM interaction
                user_msg = messages[-1]["content"] if messages[-1]["role"] == "user" else messages[-2]["content"]
                self.agent.record_llm_response(
                    prompt=user_msg,
                    response=response,
                    decision=f"Turn {turn + 1}"
                )
                
                # Check if no completion needed (can happen on first turn)
                if "NO_COMPLETION_NEEDED" in response:
                    print("Model determined no completion needed")  # Debug
                    return None, self.agent.get_tool_calls_log(), time.time() - start_time
                
                # Add assistant response to history
                messages.append({"role": "assistant", "content": response})
                
                # Parse tool calls
                tool_calls = await self._parse_tool_calls_from_response(response)
                print(f"Turn {turn + 1} - Parsed tool calls: {tool_calls}")  # Debug
                
                if not tool_calls:
                    # No tool calls found, model might be confused
                    print("No tool calls found in response")
                    break
                
                # Execute tool calls and collect results
                tool_results = []
                for tool_call in tool_calls:
                    result = await self._execute_tool_call(tool_call)
                    print(f"Tool {tool_call['tool']} result: {result}")  # Debug
                    
                    # If this is a return_answer call, we're done!
                    if tool_call["tool"] == "return_answer" and result:
                        print(f"Returning answer: {result}")  # Debug
                        return result, self.agent.get_tool_calls_log(), time.time() - start_time
                    
                    tool_results.append({
                        "tool": tool_call["tool"],
                        "args": tool_call.get("args", {}),
                        "result": result
                    })
                
                # Add tool results to conversation for next turn
                if tool_results:
                    tool_results_text = "Tool results:\n"
                    for tr in tool_results:
                        tool_results_text += f"{tr['tool']}: {tr['result']}\n"
                    messages.append({"role": "user", "content": tool_results_text})
            
            return None, self.agent.get_tool_calls_log(), time.time() - start_time
            
        except Exception as e:
            print(f"Error in get_completion: {e}")
            return None, self.agent.get_tool_calls_log(), time.time() - start_time