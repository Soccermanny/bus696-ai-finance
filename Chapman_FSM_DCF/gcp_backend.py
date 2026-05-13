import os
import json
import requests
from google.cloud import bigquery
from google.cloud import storage
from flask import jsonify

# Initialize GCP clients
bq_client = bigquery.Client()
storage_client = storage.Client()

DATASET_ID = "pib_intelligence"
TABLE_ID = "searches"
BUCKET_NAME = "financial-model"


def _fmt_billions(value):
    """Format numeric values to $XB for compact estimates table display."""
    try:
        n = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"${n / 1e9:.1f}B"


def _fmt_margin(value):
    """Format decimal margin values as percentages."""
    try:
        n = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{n * 100:.1f}%"


def _fmt_eps(value):
    """Format EPS values with a leading dollar sign."""
    try:
        n = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"${n:.2f}"


def map_estimates_payload(raw_estimates):
    """Map FMP analyst-estimates response into frontend EstimatesView shape."""
    if not isinstance(raw_estimates, list) or len(raw_estimates) == 0:
        return {
            "numAnalysts": 0,
            "years": [],
            "rows": [
                {"label": "Revenue", "values": [], "style": "bold"},
                {"label": "EBITDA Margin", "values": [], "style": "normal"},
                {"label": "EPS", "values": [], "style": "bold"}
            ]
        }

    # Keep up to the next 5 estimate periods in chronological order.
    ordered = sorted(raw_estimates, key=lambda x: x.get("date", ""))[:5]
    years = [str(item.get("date", "N/A")).split("-")[0] for item in ordered]

    revenue_values = [_fmt_billions(item.get("estimatedRevenueAvg")) for item in ordered]
    margin_values = [_fmt_margin(item.get("estimatedEbitdaAvg") / item.get("estimatedRevenueAvg"))
                     if item.get("estimatedRevenueAvg") not in (None, 0) else "N/A"
                     for item in ordered]
    eps_values = [_fmt_eps(item.get("estimatedEpsAvg")) for item in ordered]

    # Use the first available analyst coverage count from revenue or EPS fields.
    num_analysts = 0
    for item in ordered:
        candidate = item.get("numberAnalystEstimatedRevenue") or item.get("numberAnalystsEstimatedEps")
        if isinstance(candidate, (int, float)):
            num_analysts = int(candidate)
            break

    return {
        "numAnalysts": num_analysts,
        "years": years,
        "rows": [
            {"label": "Revenue", "values": revenue_values, "style": "bold"},
            {"label": "EBITDA Margin", "values": margin_values, "style": "normal"},
            {"label": "EPS", "values": eps_values, "style": "bold"}
        ]
    }

def fetch_financials(request):
    """Cloud Function to proxy FMP API, log to BigQuery, and handle GCS storage"""
    
    # Set CORS headers
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)

    headers = {'Access-Control-Allow-Origin': '*'}
    
    request_json = request.get_json(silent=True)
    if not request_json or 'ticker' not in request_json:
        return (jsonify({"error": "Missing ticker"}), 400, headers)
    
    ticker = request_json['ticker'].upper()
    section = request_json.get('section', 'snapshot')
    action = request_json.get('action', 'fetch') # 'fetch' or 'store'
    fmp_key = os.environ.get('FMP_API_KEY')
    
    try:
        if action == 'store':
            # ─── Store to GCS ───
            content = request_json.get('content')
            filename = f"exports/{ticker}_{section}_{os.urandom(4).hex()}.json"
            bucket = storage_client.bucket(BUCKET_NAME)
            blob = bucket.blob(filename)
            blob.upload_from_string(json.dumps(content), content_type='application/json')
            return (jsonify({"message": "Stored in GCS", "path": filename}), 200, headers)

        # ─── API Logic (Fetch) ───
        base_url = "https://financialmodelingprep.com/api/v3"
        endpoints = {
            "snapshot": f"{base_url}/profile/{ticker}?apikey={fmp_key}",
            "quote": f"{base_url}/quote/{ticker}?apikey={fmp_key}",
            "income": f"{base_url}/income-statement/{ticker}?limit=5&apikey={fmp_key}",
            "balance": f"{base_url}/balance-sheet-statement/{ticker}?limit=5&apikey={fmp_key}",
            "cashflow": f"{base_url}/cash-flow-statement/{ticker}?limit=5&apikey={fmp_key}",
            "valuation": f"{base_url}/key-metrics-ttm/{ticker}?apikey={fmp_key}",
            "ratios": f"{base_url}/ratios-ttm/{ticker}?apikey={fmp_key}",
            "estimates": f"{base_url}/analyst-estimates/{ticker}?limit=5&apikey={fmp_key}",
            "news": f"{base_url}/stock_news?tickers={ticker}&limit=5&apikey={fmp_key}",
            "enterprise": f"{base_url}/enterprise-values/{ticker}?limit=1&apikey={fmp_key}"
        }

        target_url = endpoints.get(section)
        if not target_url:
            return (jsonify({"error": "Invalid section"}), 400, headers)

        # Handle Snapshots by fetching both Profile and Quote
        if section == "snapshot":
            profile_res = requests.get(endpoints["snapshot"])
            quote_res = requests.get(endpoints["quote"])
            data = {
                "profile": profile_res.json(),
                "quote": quote_res.json()
            }
        elif section == "estimates":
            response = requests.get(target_url)
            data = map_estimates_payload(response.json())
        else:
            response = requests.get(target_url)
            data = response.json()
        
        # ─── Log to BigQuery ───
        # Note: Table schema must match or it will fail
        # table_ref = f"{bq_client.project}.{DATASET_ID}.{TABLE_ID}"
        # bq_client.insert_rows_json(table_ref, [{"ticker": ticker, "section": section, "data": json.dumps(data)[:1000]}])

        return (jsonify(data), 200, headers)
        
    except Exception as e:
        return (jsonify({"error": str(e)}), 500, headers)
        