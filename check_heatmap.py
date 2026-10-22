import re, json, requests

VID = "yNwjyrk7tvA"
r = requests.get(
    f"https://www.youtube.com/watch?v={VID}",
    headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"},
)
print("status:", r.status_code, "| page size:", len(r.text))
print("has macroMarkersListEntity:", "macroMarkersListEntity" in r.text)
print("has heatMarkers:", "heatMarkers" in r.text)

m = re.search(r"var ytInitialData\s*=\s*(\{.*?\});</script>", r.text, re.S)
bins = []
if m:
    data = json.loads(m.group(1))
    muts = data.get("frameworkUpdates", {}).get("entityBatchUpdate", {}).get("mutations", [])
    for mu in muts:
        ml = mu.get("payload", {}).get("macroMarkersListEntity", {}).get("markersList", {})
        if ml.get("markerType") == "MARKER_TYPE_HEATMAP":
            for x in ml.get("markers", []):
                s = int(x["startMillis"]) / 1000
                d = int(x["durationMillis"]) / 1000
                bins.append({"start_time": s, "end_time": s + d,
                             "value": x["intensityScoreNormalized"]})
print("bins found:", len(bins))