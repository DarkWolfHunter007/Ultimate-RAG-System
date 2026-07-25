import httpx

r = httpx.get("http://127.0.0.1:8000/api/config")
print("CONFIG HTTP STATUS:", r.status_code)

r_q = httpx.post("http://127.0.0.1:8000/api/query", json={"query": "What is this system?"})
print("QUERY HTTP STATUS:", r_q.status_code)
data = r_q.json()
print("QUERY MODEL:", data.get("model"))
print("QUERY ANSWER SNIPPET:", data.get("answer", "")[:100].encode('ascii', 'ignore').decode('ascii'))
