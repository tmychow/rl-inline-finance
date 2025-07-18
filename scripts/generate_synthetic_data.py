import asyncio
import random
from typing import List, Dict, Optional
import pandas as pd
import os
import sys

# Allow running the script directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import database helpers
from backend.database import (
    get_all_tickers,
    get_all_metrics,
    get_available_periods,
    get_financial_value,
    get_db,
)

# Formatting helpers

def format_value(value: float, unit: str) -> str:
    if unit == "USD_billions":
        return f"${value:.1f} billion"
    if unit == "USD":
        return f"${value:.2f}"
    if unit == "percentage":
        return f"{value:.1f}%"
    if unit == "count":
        return f"{int(round(value)):,}"
    if unit == "ratio":
        return f"{value:.2f}"
    return str(value)

async def get_company_name(ticker: str) -> str:
    async with get_db() as db:
        async with db.execute(
            "SELECT company_name FROM tickers WHERE ticker = ?",
            (ticker,)
        ) as cur:
            row = await cur.fetchone()
            return row["company_name"] if row and row["company_name"] else ticker

# Generation helpers
async def random_period(ticker: str, metric: str) -> Optional[str]:
    periods = await get_available_periods(ticker, metric)
    return random.choice(periods) if periods else None

async def generate_simple_case(tickers: List[str], metrics: List[Dict[str, str]]):
    for _ in range(10):
        ticker = random.choice(tickers)
        metric = random.choice(metrics)
        period = await random_period(ticker, metric["metric_name"])
        if not period:
            continue
        value = await get_financial_value(ticker, metric["metric_name"], period)
        if not value:
            continue
        company = await get_company_name(ticker)
        prefix_templates = [
            "{company}'s {desc} in {period} was ",
            "The {desc} for {company} in {period} was ",
            "In {period}, {company}'s {desc} was ",
        ]
        prefix = random.choice(prefix_templates).format(
            company=company,
            desc=metric["description"],
            period=period,
        )
        completion = format_value(value["value"], value.get("unit"))
        return {"input": prefix, "ground_truth": completion}
    return None

async def generate_difference_case(tickers: List[str], metrics: List[Dict[str, str]]):
    for _ in range(10):
        ticker = random.choice(tickers)
        metric = random.choice(metrics)
        periods = await get_available_periods(ticker, metric["metric_name"])
        if len(periods) < 2:
            continue
        p1, p2 = sorted(random.sample(periods, 2))
        v1 = await get_financial_value(ticker, metric["metric_name"], p1)
        v2 = await get_financial_value(ticker, metric["metric_name"], p2)
        if not v1 or not v2:
            continue
        diff = v2["value"] - v1["value"]
        company = await get_company_name(ticker)
        prefix = f"The change in {metric['description']} for {company} from {p1} to {p2} is "
        completion = format_value(diff, v1.get("unit"))
        return {"input": prefix, "ground_truth": completion}
    return None

async def generate_cagr_case(tickers: List[str], metrics: List[Dict[str, str]]):
    # Only use FY periods for CAGR
    for _ in range(10):
        ticker = random.choice(tickers)
        metric = random.choice(metrics)
        periods = [p for p in await get_available_periods(ticker, metric["metric_name"]) if p.endswith("FY")]
        if len(periods) < 2:
            continue
        start, end = sorted(random.sample(periods, 2))
        v_start = await get_financial_value(ticker, metric["metric_name"], start)
        v_end = await get_financial_value(ticker, metric["metric_name"], end)
        if not v_start or not v_end:
            continue
        years = int(end[:4]) - int(start[:4])
        if years <= 0:
            continue
        if v_start["value"] == 0 or v_end["value"] == 0:
            continue
        cagr = ((v_end["value"] / v_start["value"]) ** (1 / years) - 1) * 100
        company = await get_company_name(ticker)
        prefix = f"The CAGR of {metric['description']} for {company} from {start} to {end} is "
        completion = f"{cagr:.1f}%"
        return {"input": prefix, "ground_truth": completion}
    return None

async def generate_cross_ticker_difference_case(
    tickers: List[str], metrics: List[Dict[str, str]]
):
    for _ in range(10):
        metric = random.choice(metrics)
        t1, t2 = random.sample(tickers, 2)
        periods1 = await get_available_periods(t1, metric["metric_name"])
        periods2 = await get_available_periods(t2, metric["metric_name"])
        common = list(set(periods1).intersection(periods2))
        if not common:
            continue
        period = random.choice(common)
        v1 = await get_financial_value(t1, metric["metric_name"], period)
        v2 = await get_financial_value(t2, metric["metric_name"], period)
        if not v1 or not v2:
            continue
        company1 = await get_company_name(t1)
        company2 = await get_company_name(t2)
        diff = abs(v1["value"] - v2["value"])
        prefix = (
            f"The difference in {metric['description']} between {company1} and {company2} in {period} was "
        )
        completion = format_value(diff, v1.get("unit"))
        return {"input": prefix, "ground_truth": completion}
    return None


CALC_COMBOS = [
    {
        "m1": "netinc",
        "m2": "revenue",
        "operation": "divide",
        "unit": "percentage",
        "description": "net profit margin",
    },
    {
        "m1": "grossProfit",
        "m2": "revenue",
        "operation": "divide",
        "unit": "percentage",
        "description": "gross margin",
    },
    {
        "m1": "debt",
        "m2": "equity",
        "operation": "divide",
        "unit": "ratio",
        "description": "debt to equity ratio",
    },
    {
        "m1": "revenue",
        "m2": "costRev",
        "operation": "subtract",
        "unit": "USD_billions",
        "description": "gross profit (calc)",
    },
    {
        "m1": "assetsCurrent",
        "m2": "liabilitiesCurrent",
        "operation": "subtract",
        "unit": "USD_billions",
        "description": "working capital",
    },
]


async def generate_multi_metric_calc_case(
    tickers: List[str], metrics: List[Dict[str, str]]
):
    for _ in range(10):
        ticker = random.choice(tickers)
        combo = random.choice(CALC_COMBOS)
        periods1 = await get_available_periods(ticker, combo["m1"])
        periods2 = await get_available_periods(ticker, combo["m2"])
        common = list(set(periods1).intersection(periods2))
        if not common:
            continue
        period = random.choice(common)
        v1 = await get_financial_value(ticker, combo["m1"], period)
        v2 = await get_financial_value(ticker, combo["m2"], period)
        if not v1 or not v2:
            continue
        if combo["operation"] == "divide":
            if v2["value"] == 0:
                continue
            result = v1["value"] / v2["value"]
            if combo["unit"] == "percentage":
                result *= 100
        else:
            result = v1["value"] - v2["value"]
        company = await get_company_name(ticker)
        prefix = (
            f"The {combo['description']} for {company} in {period} was "
        )
        completion = format_value(result, combo["unit"])
        return {"input": prefix, "ground_truth": completion}
    return None

STATIC_NO_COMPLETION_PREFIXES = [
    "The CFO mentioned during the call that ",
    "This quarter the company expects that ",
    "Financial analysts often say that ",
    "The board announced today that ",
    "According to the press release, ",
    "During the investor day they said that ",
    "The CEO remarked in the interview that ",
    "Analysts on Wall Street are predicting ",
    "Management highlighted in the annual report that ",
    "In recent news articles it was reported that ",
    "Industry insiders often claim that ",
    "The marketing team announced that ",
]

DYNAMIC_NO_COMPLETION_TEMPLATES = [
    "{company}'s {desc} can be broken down into a few business lines",
    "Analysts often track {company}'s {desc} closely",
    "During the {period} call, management discussed {company}'s {desc}",
    "{company}'s team noted that its {desc} trend is improving",
    "There has been speculation about how {company}'s {desc} might change",
]

async def generate_no_completion_case(
    tickers: List[str], metrics: List[Dict[str, str]]
) -> Dict[str, str]:
    if random.random() < 0.5:
        prefix = random.choice(STATIC_NO_COMPLETION_PREFIXES)
    else:
        ticker = random.choice(tickers)
        metric = random.choice(metrics)
        period = await random_period(ticker, metric["metric_name"])
        company = await get_company_name(ticker)
        template = random.choice(DYNAMIC_NO_COMPLETION_TEMPLATES)
        prefix = template.format(company=company, desc=metric["description"], period=period or "the last quarter")
    return {"input": prefix, "ground_truth": "NO_COMPLETION_NEEDED"}

async def generate_cases(num_cases: int) -> List[Dict[str, str]]:
    tickers = await get_all_tickers()
    metrics = await get_all_metrics()
    cases = []
    generators = [
        generate_simple_case,
        generate_difference_case,
        generate_cross_ticker_difference_case,
        generate_multi_metric_calc_case,
        generate_cagr_case,
    ]
    weights = [0.4, 0.2, 0.1, 0.15, 0.15]

    while len(cases) < num_cases:
        if random.random() < 0.25:
            case = await generate_no_completion_case(tickers, metrics)
            cases.append(case)
            continue
        gen = random.choices(generators, weights)[0]
        case = await gen(tickers, metrics)
        if case:
            cases.append(case)
    return cases

async def main():
    eval_cases = await generate_cases(50)
    train_cases = await generate_cases(200)

    pd.DataFrame(eval_cases).to_csv("data/evaluation_test_cases.csv", index=False)
    pd.DataFrame(train_cases).to_csv("data/training_test_cases.csv", index=False)
    print("Generated evaluation_test_cases.csv and training_test_cases.csv")

if __name__ == "__main__":
    asyncio.run(main())
