"""
utils/db.py – SQLite database layer for VigiRoad AI
─────────────────────────────────────────────────────────
Alert threshold logic:
  damage_rate >= 80  → SEVERE  → alert sent to Police + Ambulance
  damage_rate 50–79  → MEDIUM  → alert sent to Police + Ambulance
  damage_rate < 50   → LOW     → displayed only, NO alert
  Alert is recorded AFTER the accident is logged (not before).
"""
import sqlite3, os, hashlib, math, random
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "vigiroad.db")


def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    c = conn.cursor()

    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'USER',
        station_name TEXT DEFAULT NULL,
        station_lat  REAL DEFAULT NULL,
        station_lon  REAL DEFAULT NULL,
        coverage_km  REAL DEFAULT 10.0
    )""")

    for col, defn in [
        ("station_name", "TEXT DEFAULT NULL"),
        ("station_lat",  "REAL DEFAULT NULL"),
        ("station_lon",  "REAL DEFAULT NULL"),
        ("coverage_km",  "REAL DEFAULT 10.0"),
    ]:
        try:
            c.execute(f"ALTER TABLE users ADD COLUMN {col} {defn}")
        except sqlite3.OperationalError:
            pass

    c.execute("""CREATE TABLE IF NOT EXISTS reports (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        type          TEXT NOT NULL,
        severity      TEXT NOT NULL,
        location_name TEXT,
        lat           REAL,
        lon           REAL,
        status        TEXT DEFAULT 'pending',
        assigned_to   TEXT DEFAULT NULL,
        timestamp     TEXT NOT NULL,
        damage_rate   INTEGER DEFAULT 0,
        alert_sent    INTEGER DEFAULT 0,
        alert_time    TEXT DEFAULT NULL
    )""")

    for col, defn in [
        ("damage_rate", "INTEGER DEFAULT 0"),
        ("alert_sent",  "INTEGER DEFAULT 0"),
        ("alert_time",  "TEXT DEFAULT NULL"),
    ]:
        try:
            c.execute(f"ALTER TABLE reports ADD COLUMN {col} {defn}")
        except sqlite3.OperationalError:
            pass

    conn.commit()

    # Normalise legacy severity labels
    c.execute("UPDATE reports SET severity='medium' WHERE severity='moderate'")
    c.execute("UPDATE reports SET severity='severe' WHERE severity IN ('high','critical')")
    c.execute("UPDATE reports SET severity='low'    WHERE severity IN ('minor','minimal')")

    # Back-fill damage_rate for old accident rows
    c.execute("""UPDATE reports SET damage_rate = CASE severity
        WHEN 'severe' THEN 85 WHEN 'medium' THEN 62 ELSE 28
        END WHERE damage_rate = 0 AND type = 'accident'""")

    # Back-fill alert_sent for old rows that qualify
    c.execute("""UPDATE reports SET
        alert_sent = 1, alert_time = timestamp
        WHERE alert_sent = 0 AND damage_rate >= 50 AND type = 'accident'""")

    conn.commit()
    conn.close()


def _hash(pwd):
    return hashlib.sha256(pwd.encode()).hexdigest()


def authenticate(email, password):
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM users WHERE email=? AND password=?",
        (email, _hash(password))
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def register_user(name, email, password, role="USER",
                  station_name=None, station_lat=None, station_lon=None, coverage_km=10.0):
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO users (name,email,password,role,station_name,station_lat,station_lon,coverage_km) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (name, email, _hash(password), role, station_name, station_lat, station_lon, coverage_km)
        )
        conn.commit()
        return True, "Account created successfully!"
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."
    finally:
        conn.close()


_SEV_NORM = {
    "moderate": "medium", "high": "severe", "critical": "severe",
    "minor": "low", "minimal": "low",
}


def _derive_damage_rate(severity: str) -> int:
    if severity == "severe":
        return random.randint(80, 98)
    elif severity == "medium":
        return random.randint(50, 79)
    return random.randint(5, 49)


def should_alert(damage_rate: int) -> bool:
    return damage_rate >= 50


def alert_level(damage_rate: int) -> str:
    if damage_rate >= 80:   return "SEVERE"
    elif damage_rate >= 50: return "MEDIUM"
    return "LOW"


def add_report(type_: str, severity: str, location_name: str,
               lat: float, lon: float, damage_rate: int = None):
    severity    = _SEV_NORM.get(severity, severity)
    now         = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    damage_rate = damage_rate if damage_rate is not None else _derive_damage_rate(severity)
    do_alert    = 1 if (type_ == "accident" and should_alert(damage_rate)) else 0
    alert_time  = now if do_alert else None

    conn = get_conn()
    conn.execute(
        "INSERT INTO reports "
        "(type,severity,location_name,lat,lon,timestamp,damage_rate,alert_sent,alert_time) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (type_, severity, location_name, lat, lon, now, damage_rate, do_alert, alert_time)
    )
    conn.commit()
    conn.close()


def get_reports(type_=None, status=None):
    conn = get_conn()
    q, params = "SELECT * FROM reports WHERE 1=1", []
    if type_:  q += " AND type=?";   params.append(type_)
    if status: q += " AND status=?"; params.append(status)
    q += " ORDER BY id DESC"
    rows = conn.execute(q, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _haversine_db(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2)
    return R * 2 * math.asin(math.sqrt(a))


def get_reports_in_area(lat, lon, coverage_km, type_=None, status=None):
    all_reports = get_reports(type_=type_, status=status)
    filtered = []
    for r in all_reports:
        if r.get("lat") and r.get("lon"):
            dist = _haversine_db(lat, lon, r["lat"], r["lon"])
            if dist <= coverage_km:
                r["distance_km"] = round(dist, 1)
                filtered.append(r)
        else:
            r["distance_km"] = None
            filtered.append(r)
    return filtered


def get_stats_in_area(lat=None, lon=None, coverage_km=None):
    if lat is not None and lon is not None and coverage_km is not None:
        all_r = get_reports()
        filtered = [r for r in all_r
                    if r.get("lat") and r.get("lon") and
                    _haversine_db(lat, lon, r["lat"], r["lon"]) <= coverage_km]
    else:
        filtered = get_reports()

    return dict(
        total     = len(filtered),
        accidents = sum(1 for r in filtered if r["type"] == "accident"),
        potholes  = sum(1 for r in filtered if r["type"] == "pothole"),
        pending   = sum(1 for r in filtered if r["status"] == "pending"),
        resolved  = sum(1 for r in filtered if r["status"] == "resolved"),
        severe    = sum(1 for r in filtered if r["severity"] == "severe"),
        alerts    = sum(1 for r in filtered
                        if r.get("alert_sent") and r["type"] == "accident"),
    )


def update_report_status(report_id, status):
    conn = get_conn()
    conn.execute("UPDATE reports SET status=? WHERE id=?", (status, report_id))
    conn.commit()
    conn.close()


def get_stats():
    return get_stats_in_area()
