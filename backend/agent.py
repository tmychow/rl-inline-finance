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
        get_all_tickers, get_tickers_with_names, get_all_metrics, get_financial_value,
        get_available_periods, get_latest_period
    )
except ImportError:
    from database import (
        get_all_tickers, get_tickers_with_names, get_all_metrics, get_financial_value,
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
    
    async def get_tickers(self) -> List[Dict[str, str]]:
        tickers = await get_tickers_with_names()
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
            # Allow CAGR calculation as long as the ratio (num1 / num2) is positive.
            # This handles cases where both values are negative (e.g., -0.1635 / -0.9334),
            # while still protecting against invalid scenarios such as division by zero,
            # mismatched signs, or non-positive duration.
            if num2 != 0 and duration > 0 and (num1 / num2) > 0:
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
    def __init__(self, model_provider: str = "openai", model_name: str = "gpt-4.1", hf_model=None, hf_tokenizer=None):
        self.model_provider = model_provider
        self.model_name = model_name
        self.agent = FinancialAgent()
        self.hf_model = hf_model
        self.hf_tokenizer = hf_tokenizer
    
    async def _parse_tool_calls_from_response(self, response: str) -> List[Dict[str, Any]]:
        tool_calls = []
        
        # Define function signatures and their expected parameters
        tool_signatures = {
            "get_metrics": [],
            "get_tickers": [],
            "get_value": ["metric", "ticker", "period"],
            "calculate": ["num1", "num2", "operation", "duration"],
            "return_answer": ["answer"]
        }
        
        # Find the first tool call in the response
        earliest_match = None
        earliest_pos = len(response)
        matched_tool = None
        
        # Find which tool appears first in the response
        for tool_name, params in tool_signatures.items():
            pattern = rf"{tool_name}\s*\("
            match = re.search(pattern, response)
            if match and match.start() < earliest_pos:
                earliest_match = match
                earliest_pos = match.start()
                matched_tool = (tool_name, params)
        
        # If we found a tool call, parse only that one
        if earliest_match and matched_tool:
            tool_name, params = matched_tool
            match = earliest_match
            start = match.end()
            
            # Find matching closing parenthesis
            paren_count = 1
            end = start
            while end < len(response) and paren_count > 0:
                if response[end] == '(':
                    paren_count += 1
                elif response[end] == ')':
                    paren_count -= 1
                end += 1
            
            if paren_count == 0:
                args_str = response[start:end-1].strip()
                
                # Parse arguments
                parsed_args = {}
                
                if not args_str:  # No arguments
                    if tool_name in ["get_metrics", "get_tickers"]:
                        tool_calls.append({"tool": tool_name, "args": {}})
                else:
                    # Try to parse arguments (both positional and keyword)
                    if tool_name == "return_answer":
                        # Special handling for return_answer to capture everything
                        # Remove quotes if present
                        answer = args_str
                        if answer.startswith('answer='):
                            answer = answer[7:].strip()
                        # Remove surrounding quotes
                        for quote in ['"', "'", '"""', "'''"]:
                            if answer.startswith(quote) and answer.endswith(quote):
                                answer = answer[len(quote):-len(quote)]
                                break
                        parsed_args = {"answer": answer}
                    else:
                        # Parse other functions
                        arg_values = []
                        
                        # First try to parse keyword arguments
                        has_keywords = '=' in args_str
                        
                        if has_keywords:
                            # Parse keyword arguments
                            # Split by comma but not inside quotes
                            parts = re.split(r',(?=(?:[^"\']*["\'][^"\']*["\'])*[^"\']*$)', args_str)
                            for part in parts:
                                if '=' in part:
                                    key, val = part.split('=', 1)
                                    key = key.strip()
                                    val = val.strip()
                                    # Remove quotes
                                    for quote in ['"', "'", '"""', "'''"]:
                                        if val.startswith(quote) and val.endswith(quote):
                                            val = val[len(quote):-len(quote)]
                                            break
                                    if key in params:
                                        parsed_args[key] = val
                        else:
                            # Parse positional arguments
                            # Split by comma but not inside quotes
                            parts = re.split(r',(?=(?:[^"\']*["\'][^"\']*["\'])*[^"\']*$)', args_str)
                            for i, part in enumerate(parts):
                                if i < len(params):
                                    val = part.strip()
                                    # Remove quotes
                                    for quote in ['"', "'", '"""', "'''"]:
                                        if val.startswith(quote) and val.endswith(quote):
                                            val = val[len(quote):-len(quote)]
                                            break
                                    parsed_args[params[i]] = val
                        
                    # Convert types for specific functions
                    if tool_name == "calculate" and parsed_args:
                        if "num1" in parsed_args:
                            parsed_args["num1"] = float(parsed_args["num1"])
                        if "num2" in parsed_args:
                            parsed_args["num2"] = float(parsed_args["num2"])
                        if "duration" in parsed_args:
                            # Handle 'None' string, empty string, or actual None
                            duration_val = parsed_args["duration"]
                            if duration_val and duration_val.lower() != 'none':
                                try:
                                    parsed_args["duration"] = int(duration_val)
                                except ValueError:
                                    parsed_args["duration"] = None
                            else:
                                parsed_args["duration"] = None
                    
                    if parsed_args:
                        tool_calls.append({"tool": tool_name, "args": parsed_args})
        
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

        elif self.model_provider == "hf" and self.hf_model is not None and self.hf_tokenizer is not None:
            inputs = self.hf_tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.hf_model.device)
            output = self.hf_model.generate(inputs, max_new_tokens=256, temperature=0.1)
            return self.hf_tokenizer.decode(output[0, inputs.shape[1]:], skip_special_tokens=True)

        return ""
    
    def _get_system_prompt(self) -> str:
        return '''You are a financial data assistant that completes partial text with accurate financial data.

Available tools (YOU MUST USE THESE):
- get_metrics(): returns list of available metrics
- get_tickers(): returns list of available tickers  
- get_value(metric, ticker, period): gets a specific value
  - Period format: "2024Q1" for Q1 2024, "2024FY" for fiscal year 2024, or "latest" for most recent
- calculate(num1, num2, operation, duration): operations are "add", "subtract", "multiply", "divide", "CAGR"
    - for CAGR, duration is the number of time periods
    - for CAGR, num1 is the final value and num2 is the initial value
- return_answer(answer): returns the final completion text (ONLY the completion, not the full text)

If the text doesn't need financial data, use: return_answer(answer="NO_COMPLETION_NEEDED")

IMPORTANT RULES:
1. You MUST use tools - do not provide commentary or ask questions, and do not make up values
2. This is a multi-turn conversation. After you call tools, you'll receive the results and can call more tools or return the answer
3. Use one tool call at a time. Do not nest tool calls.
4. Make sure to check the available metrics AND available tickers before calling get_value.
5. Always end with return_answer() to provide your completion, and only return the completion text, not the full sentence

EXAMPLE:
User: "The revenue for Apple in 2023 was"
You: get_metrics()
User: [list of metrics]
You: get_tickers()
User: [list of tickers]
You: get_value(metric="revenue", ticker="AAPL", period="2023FY")
User: [value]
You: return_answer(answer="[value with units]")
'''
    
    async def get_completion(self, text: str) -> Tuple[Optional[str], List[Dict[str, Any]], float]:
        start_time = time.time()
        self.agent.clear_tool_calls()
        
        # Skip empty text
        if not text.strip():
            return "", [], time.time() - start_time
        
        try:
            # Build initial conversation
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": f"Complete this text with financial data, if needed. If not, return 'NO_COMPLETION_NEEDED': '{text}'"}
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
                    return "NO_COMPLETION_NEEDED", self.agent.get_tool_calls_log(), time.time() - start_time
                
                # Add assistant response to history
                messages.append({"role": "assistant", "content": response})
                
                # Parse tool calls
                tool_calls = await self._parse_tool_calls_from_response(response)
                print(f"Turn {turn + 1} - Parsed tool calls: {tool_calls}")  # Debug
                
                if not tool_calls:
                    # No tool calls found, model might be confused or providing commentary
                    print("No tool calls found in response")
                    # Check if this looks like the model is trying to return a completion without using return_answer
                    # This could happen if the model doesn't understand the tool syntax
                    if turn == 0 and len(response.strip()) > 0:
                        # On first turn, if there's a response but no tool calls, treat it as the completion
                        print(f"Treating response as direct completion: {response}")
                        return response.strip(), self.agent.get_tool_calls_log(), time.time() - start_time
                    # Otherwise, prompt the model to use tools
                    messages.append({"role": "user", "content": "Please use the available tools to complete the task. Use return_answer(answer='your completion') to provide the final answer, or return_answer(answer='NO_COMPLETION_NEEDED') if no completion is needed. Your completion should not contain the user's original text."})
                    continue
                
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
            
            # If we've exhausted all turns without getting a return_answer, this is an error
            # Return a special error message instead of None
            error_msg = "[ERROR: Model failed to provide completion after 10 turns]"
            print(error_msg)
            return error_msg, self.agent.get_tool_calls_log(), time.time() - start_time
            
        except Exception as e:
            print(f"Error in get_completion: {e}")
            error_msg = f"[ERROR: {str(e)}]"
            return error_msg, self.agent.get_tool_calls_log(), time.time() - start_time