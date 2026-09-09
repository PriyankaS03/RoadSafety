"""
utils/emergency_services.py – Police stations, hospitals & ambulance bases
Provides location data and helper functions for emergency service coverage areas.
"""
import math

# ── Police Stations (Tamil Nadu / Coimbatore region) ──────────────────────────
POLICE_STATIONS = [
    {"name": "Gandhipuram Police Station",     "lat": 11.0180, "lon": 76.9725, "coverage_km": 8,
     "officer": "SI Rajkumar",   "contact": "+91 98765 43210", "zone": "Gandhipuram Zone"},
    {"name": "RS Puram Police Station",        "lat": 11.0050, "lon": 76.9530, "coverage_km": 7,
     "officer": "SI Meenakshi",  "contact": "+91 98765 43211", "zone": "RS Puram Zone"},
    {"name": "Saibaba Colony Police Station",   "lat": 11.0240, "lon": 76.9635, "coverage_km": 6,
     "officer": "SI Venkatesh",  "contact": "+91 98765 43212", "zone": "Saibaba Colony Zone"},
    {"name": "Peelamedu Police Station",        "lat": 11.0290, "lon": 77.0020, "coverage_km": 8,
     "officer": "SI Priya",      "contact": "+91 98765 43213", "zone": "Peelamedu Zone"},
    {"name": "Singanallur Police Station",      "lat": 10.9940, "lon": 77.0180, "coverage_km": 7,
     "officer": "SI Arun",       "contact": "+91 98765 43214", "zone": "Singanallur Zone"},
    {"name": "Sulur Police Station",           "lat": 10.9930, "lon": 77.1240, "coverage_km": 10,
     "officer": "SI Divya",      "contact": "+91 98765 43215", "zone": "Sulur Zone"},
    {"name": "Mettupalayam Police Station",    "lat": 11.2990, "lon": 76.9360, "coverage_km": 12,
     "officer": "SI Karthik",    "contact": "+91 98765 43216", "zone": "Mettupalayam Zone"},
    {"name": "Tirupur Town Police Station",    "lat": 11.1085, "lon": 77.3411, "coverage_km": 10,
     "officer": "SI Lakshmi",    "contact": "+91 98765 43217", "zone": "Tirupur Town Zone"},
    {"name": "Erode Town Police Station",      "lat": 11.3410, "lon": 77.7172, "coverage_km": 12,
     "officer": "SI Murugan",    "contact": "+91 98765 43218", "zone": "Erode Town Zone"},
    {"name": "Salem Town Police Station",      "lat": 11.6643, "lon": 78.1460, "coverage_km": 12,
     "officer": "SI Deepa",      "contact": "+91 98765 43219", "zone": "Salem Town Zone"},
]

# ── Hospitals & Ambulance Bases ───────────────────────────────────────────────
HOSPITALS = [
    {"name": "Coimbatore Medical College Hospital", "lat": 11.0160, "lon": 76.9700, "coverage_km": 10,
     "type": "Government Hospital", "ambulances": 4, "beds": 250, "contact": "+91 422 230 1393",
     "speciality": "Trauma & Emergency"},
    {"name": "GKNM Hospital",                       "lat": 11.0130, "lon": 76.9520, "coverage_km": 8,
     "type": "Private Hospital",  "ambulances": 3, "beds": 180, "contact": "+91 422 215 5100",
     "speciality": "Multi-Speciality"},
    {"name": "PSG Hospitals",                        "lat": 11.0240, "lon": 77.0020, "coverage_km": 9,
     "type": "Private Hospital",  "ambulances": 3, "beds": 200, "contact": "+91 422 257 0170",
     "speciality": "Trauma Centre"},
    {"name": "Sri Ramakrishna Hospital",             "lat": 11.0150, "lon": 76.9630, "coverage_km": 8,
     "type": "Private Hospital",  "ambulances": 2, "beds": 150, "contact": "+91 422 439 8888",
     "speciality": "Emergency & ICU"},
    {"name": "KG Hospital",                          "lat": 11.0060, "lon": 76.9580, "coverage_km": 7,
     "type": "Private Hospital",  "ambulances": 2, "beds": 120, "contact": "+91 422 221 1011",
     "speciality": "Multi-Speciality"},
    {"name": "Kovai Medical Center & Hospital",      "lat": 10.9970, "lon": 76.9610, "coverage_km": 10,
     "type": "Private Hospital",  "ambulances": 4, "beds": 300, "contact": "+91 422 428 8888",
     "speciality": "Super Speciality"},
    {"name": "Mettupalayam GH",                     "lat": 11.2950, "lon": 76.9380, "coverage_km": 12,
     "type": "Government Hospital", "ambulances": 2, "beds": 80, "contact": "+91 4254 222 110",
     "speciality": "General & Trauma"},
    {"name": "Tirupur GH",                          "lat": 11.1060, "lon": 77.3350, "coverage_km": 10,
     "type": "Government Hospital", "ambulances": 3, "beds": 150, "contact": "+91 421 222 0444",
     "speciality": "General & Emergency"},
    {"name": "Erode GH",                            "lat": 11.3380, "lon": 77.7200, "coverage_km": 12,
     "type": "Government Hospital", "ambulances": 3, "beds": 200, "contact": "+91 424 222 5190",
     "speciality": "Trauma & General"},
    {"name": "Salem GH",                            "lat": 11.6700, "lon": 78.1500, "coverage_km": 12,
     "type": "Government Hospital", "ambulances": 4, "beds": 300, "contact": "+91 427 231 6688",
     "speciality": "Multi-Speciality"},
]

# ── 108 Ambulance Bases (separate from hospital-attached ambulances) ──────────
AMBULANCE_BASES = [
    {"name": "108 Ambulance – Gandhipuram",   "lat": 11.0190, "lon": 76.9730, "coverage_km": 8,
     "vehicle_no": "TN-38-AM-0108", "status": "available"},
    {"name": "108 Ambulance – Singanallur",   "lat": 10.9950, "lon": 77.0200, "coverage_km": 8,
     "vehicle_no": "TN-38-AM-0209", "status": "available"},
    {"name": "108 Ambulance – Saravanampatti","lat": 11.0560, "lon": 77.0190, "coverage_km": 10,
     "vehicle_no": "TN-38-AM-0310", "status": "available"},
    {"name": "108 Ambulance – Sulur",         "lat": 10.9950, "lon": 77.1250, "coverage_km": 10,
     "vehicle_no": "TN-38-AM-0411", "status": "available"},
    {"name": "108 Ambulance – Mettupalayam",  "lat": 11.2960, "lon": 76.9370, "coverage_km": 12,
     "vehicle_no": "TN-38-AM-0512", "status": "available"},
]


def _haversine(lat1, lon1, lat2, lon2):
    """Distance in km between two points."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def get_nearest_police(lat, lon, max_results=3):
    """Return the nearest police stations to a given coordinate, sorted by distance."""
    results = []
    for ps in POLICE_STATIONS:
        dist = _haversine(lat, lon, ps["lat"], ps["lon"])
        entry = {**ps, "distance_km": round(dist, 1)}
        results.append(entry)
    results.sort(key=lambda x: x["distance_km"])
    return results[:max_results]


def get_police_covering(lat, lon):
    """Return all police stations whose coverage area includes this point."""
    covering = []
    for ps in POLICE_STATIONS:
        dist = _haversine(lat, lon, ps["lat"], ps["lon"])
        if dist <= ps["coverage_km"]:
            covering.append({**ps, "distance_km": round(dist, 1)})
    covering.sort(key=lambda x: x["distance_km"])
    return covering


def get_nearest_hospitals(lat, lon, max_results=3):
    """Return the nearest hospitals sorted by distance."""
    results = []
    for h in HOSPITALS:
        dist = _haversine(lat, lon, h["lat"], h["lon"])
        results.append({**h, "distance_km": round(dist, 1)})
    results.sort(key=lambda x: x["distance_km"])
    return results[:max_results]


def get_hospitals_covering(lat, lon):
    """Return all hospitals whose coverage area includes this point."""
    covering = []
    for h in HOSPITALS:
        dist = _haversine(lat, lon, h["lat"], h["lon"])
        if dist <= h["coverage_km"]:
            covering.append({**h, "distance_km": round(dist, 1)})
    covering.sort(key=lambda x: x["distance_km"])
    return covering


def get_nearest_ambulances(lat, lon, max_results=2):
    """Return the nearest available ambulance bases."""
    results = []
    for a in AMBULANCE_BASES:
        dist = _haversine(lat, lon, a["lat"], a["lon"])
        results.append({**a, "distance_km": round(dist, 1),
                        "eta_min": round(dist / 40 * 60)})  # ~40 km/h urban speed
    results.sort(key=lambda x: x["distance_km"])
    return results[:max_results]
