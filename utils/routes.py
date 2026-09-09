"""
utils/routes.py – Safe route calculation engine
Uses OpenStreetMap Nominatim for geocoding and OSRM for real routing.
Safety scoring is powered by historical accident/pothole data (historical_data.py)
combined with real-time DB reports near the route.
"""
import random, math, requests
from utils.db import get_reports
from utils.historical_data import score_route_safety

# ── Geocoding via Nominatim (free, no API key) ──────────────────────────────
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

def geocode(place: str):
    """Return (lat, lon, display_name) or None."""
    try:
        resp = requests.get(NOMINATIM_URL, params={
            "q": place,
            "format": "json",
            "limit": 1,
        }, headers={"User-Agent": "VigiRoadAI/1.0"}, timeout=8)
        data = resp.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"]), data[0]["display_name"]
    except Exception:
        pass
    return None


# ── Routing via OSRM (free public demo server) ──────────────────────────────
OSRM_URL = "https://router.project-osrm.org/route/v1/driving"

def _fetch_osrm_routes(lat1, lon1, lat2, lon2):
    """
    Fetch up to 3 alternative routes from OSRM.
    Returns list of dicts: [{distance_km, duration_min, geometry_coords, waypoint_names}, ...]
    """
    try:
        url = f"{OSRM_URL}/{lon1},{lat1};{lon2},{lat2}"
        resp = requests.get(url, params={
            "alternatives": "true",
            "overview": "full",
            "geometries": "geojson",
            "steps": "true",
        }, headers={"User-Agent": "VigiRoadAI/1.0"}, timeout=12)
        data = resp.json()
        if data.get("code") != "Ok":
            return []

        results = []
        for route in data["routes"][:3]:
            # Extract key road names from steps
            road_names = []
            for leg in route["legs"]:
                for step in leg["steps"]:
                    name = step.get("name", "").strip()
                    if name and name not in road_names:
                        road_names.append(name)

            coords = route["geometry"]["coordinates"]  # [lon, lat] pairs
            results.append({
                "distance_km": round(route["distance"] / 1000, 1),
                "duration_min": round(route["duration"] / 60),
                "geometry": [(c[1], c[0]) for c in coords],  # flip to [lat, lon]
                "road_names": road_names[:8],  # keep top 8 road names
            })
        return results
    except Exception:
        return []


def _haversine(lat1, lon1, lat2, lon2):
    """Distance in km between two points."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def _count_hazards_near_route(geometry_coords, radius_km=2.0):
    """Count DB reports that fall within `radius_km` of any point on the route."""
    reports = get_reports()
    accidents = 0
    potholes = 0
    # Sample every Nth point to keep it fast
    step = max(1, len(geometry_coords) // 50)
    sampled = geometry_coords[::step]

    for r in reports:
        if not r.get("lat") or not r.get("lon"):
            continue
        for pt in sampled:
            if _haversine(pt[0], pt[1], r["lat"], r["lon"]) <= radius_km:
                if r["type"] == "accident":
                    accidents += 1
                else:
                    potholes += 1
                break  # count each report only once
    return accidents, potholes


def _safety_label(score):
    if score <= 2.0:  return "Low Risk Road",   "✅", "#10b981", "route-safe"
    elif score <= 4.5: return "Medium Risk Road", "⚠️", "#f59e0b", "route-medium"
    else:              return "High Risk Road",  "🔴", "#ef4444", "route-risky"


def get_routes(origin_coords, dest_coords, origin_name, dest_name):
    """
    Fetch real routes between two geocoded points.
    Returns a list of route dicts sorted safest-first.

    Safety scoring uses:
    1. Historical accident/pothole data per road segment
    2. Real-time DB incident reports near the route
    3. AI-weighted model (accident freq 40%, potholes 25%, condition 20%, road type 15%)
    """
    lat1, lon1 = origin_coords
    lat2, lon2 = dest_coords
    osrm_routes = _fetch_osrm_routes(lat1, lon1, lat2, lon2)

    # Fallback: if OSRM fails, generate a synthetic straight-line route
    if not osrm_routes:
        dist = round(_haversine(lat1, lon1, lat2, lon2), 1)
        osrm_routes = [{
            "distance_km": dist,
            "duration_min": round(dist / 40 * 60),  # assume ~40 km/h
            "geometry": [(lat1, lon1), (lat2, lon2)],
            "road_names": ["Direct route"],
        }]

    routes = []
    for i, osrm in enumerate(osrm_routes):
        # Count real-time hazards from database
        rt_accidents, rt_potholes = _count_hazards_near_route(osrm["geometry"])

        # Score using AI historical model
        hist_score, hist_accidents, hist_potholes, road_quality = score_route_safety(
            osrm["road_names"], osrm["distance_km"]
        )

        # Combine: historical model (70%) + real-time DB (30%)
        rt_score = rt_accidents * 0.6 + rt_potholes * 0.4
        combined_score = round(hist_score * 0.7 + rt_score * 0.3, 1)

        # Total counts (historical + real-time)
        total_accidents = hist_accidents + rt_accidents
        total_potholes = hist_potholes + rt_potholes

        label, emoji, color, css = _safety_label(combined_score)

        # Build via string from actual road names (like the reference image)
        via_str = " → ".join(osrm["road_names"][:6]) if osrm["road_names"] else "Direct"

        routes.append({
            "name":         via_str,  # Use actual road names as route name
            "km":           osrm["distance_km"],
            "minutes":      osrm["duration_min"],
            "via":          via_str,
            "road_names":   osrm["road_names"],
            "accidents":    total_accidents,
            "potholes":     total_potholes,
            "score":        combined_score,
            "road_quality": road_quality,
            "label":        label,
            "emoji":        emoji,
            "color":        color,
            "css":          css,
            "geometry":     osrm["geometry"],
        })

    routes.sort(key=lambda r: r["score"])
    return routes
