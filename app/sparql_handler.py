# sparql_handler.py
import requests
import time
from typing import List, Union, Optional

# Configure these fields to identify your client to Wikimedia or other endpoints.
# Replace the contact string with an email or project URL if you plan to use public Wikidata.
HEADERS = {
    "Accept": "application/sparql-results+json",
    "User-Agent": "MultimodeApp/1.0 (contact: user@example.com)"
}


def query_sparql(endpoint_url: str, query: str, timeout: int = 30, retries: int = 3, backoff_base: int = 2) -> List[Union[str, tuple]]:
    """
    Execute a SPARQL query and return results as a list of strings or tuples.
    Retries on 429 and on network read timeouts with exponential backoff.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            response = requests.get(endpoint_url, params={"query": query}, headers=HEADERS, timeout=timeout)
            if response.status_code == 429:
                # Rate limited — back off and retry
                wait = backoff_base ** attempt
                time.sleep(wait)
                last_exc = RuntimeError("429")
                continue
            response.raise_for_status()
            data = response.json()

            results = []
            vars_ = data.get("head", {}).get("vars", [])
            for item in data.get("results", {}).get("bindings", []):
                row = []
                for v in vars_:
                    if v in item:
                        row.append(item[v]["value"])
                    else:
                        row.append(None)
                if len(row) == 1:
                    results.append(row[0])
                else:
                    results.append(tuple(row))
            return results
        except requests.exceptions.ReadTimeout as e:
            last_exc = e
            wait = backoff_base ** attempt
            time.sleep(wait)
            continue
        except requests.exceptions.RequestException as e:
            # For other HTTP/network errors, raise after last attempt
            last_exc = e
            break

    # If we get here, all attempts failed
    if last_exc:
        raise last_exc
    return []
