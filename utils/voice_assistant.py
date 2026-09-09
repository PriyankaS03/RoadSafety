"""
utils/voice_assistant.py
Injects Web Speech API + Audio beep sounds into Streamlit pages.
All speech/sound runs client-side via JavaScript — no server load.
"""


def _beep_js(freq=880, duration=0.18, volume=0.7, wave="square") -> str:
    """Generate JS snippet that plays a single beep tone."""
    return (
        f"(function(){{var a=new(window.AudioContext||window.webkitAudioContext)();"
        f"var o=a.createOscillator();var g=a.createGain();"
        f"o.type='{wave}';o.frequency.value={freq};"
        f"g.gain.setValueAtTime({volume},a.currentTime);"
        f"g.gain.exponentialRampToValueAtTime(0.001,a.currentTime+{duration});"
        f"o.connect(g);g.connect(a.destination);"
        f"o.start(a.currentTime);o.stop(a.currentTime+{duration});}})();"
    )


def _alert_beep_js() -> str:
    """Two-tone urgent alert beep (police/ambulance alert)."""
    return (
        "(function(){"
        "var a=new(window.AudioContext||window.webkitAudioContext)();"
        "function beep(f,t,d){"
        "var o=a.createOscillator();var g=a.createGain();"
        "o.type='square';o.frequency.value=f;"
        "g.gain.setValueAtTime(0.6,a.currentTime+t);"
        "g.gain.exponentialRampToValueAtTime(0.001,a.currentTime+t+d);"
        "o.connect(g);g.connect(a.destination);"
        "o.start(a.currentTime+t);o.stop(a.currentTime+t+d);}"
        "beep(880,0,0.15);beep(660,0.18,0.15);beep(880,0.36,0.15);beep(660,0.54,0.15);"
        "})();"
    )


def _pothole_beep_js() -> str:
    """Single low-pitch bump sound for pothole warning."""
    return _beep_js(freq=280, duration=0.25, volume=0.55, wave="sawtooth")


def _hospital_beep_js() -> str:
    """Gentle ascending chime for hospital proximity."""
    return (
        "(function(){"
        "var a=new(window.AudioContext||window.webkitAudioContext)();"
        "function b(f,t){"
        "var o=a.createOscillator();var g=a.createGain();"
        "o.type='sine';o.frequency.value=f;"
        "g.gain.setValueAtTime(0.4,a.currentTime+t);"
        "g.gain.exponentialRampToValueAtTime(0.001,a.currentTime+t+0.3);"
        "o.connect(g);g.connect(a.destination);"
        "o.start(a.currentTime+t);o.stop(a.currentTime+t+0.35);}"
        "b(523,0);b(659,0.2);b(784,0.4);"
        "})();"
    )


def _speak_js(text: str, rate: float = 0.92, pitch: float = 1.05) -> str:
    """Generate JS to speak text via Web Speech API."""
    # Escape single quotes in text
    safe = text.replace("'", "\\'").replace('"', '\\"').replace("\n", " ")
    return (
        f"(function(){{"
        f"if(!window.speechSynthesis){{return;}}"
        f"window.speechSynthesis.cancel();"
        f"var u=new SpeechSynthesisUtterance('{safe}');"
        f"u.rate={rate};u.pitch={pitch};u.volume=0.95;u.lang='en-IN';"
        f"window.speechSynthesis.speak(u);"
        f"}})();"
    )


def _combined_js(*snippets: str) -> str:
    return "".join(snippets)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC HELPERS  (called from dashboard pages)
# ─────────────────────────────────────────────────────────────────────────────

def voice_alert_widget_html(enabled_key: str = "va_enabled") -> str:
    """
    Returns the full HTML/JS for the persistent voice assistant widget
    that floats in the bottom-right corner.
    Call st.markdown(voice_alert_widget_html(), unsafe_allow_html=True) once per page.
    """
    return """
<style>
#va-widget {
    position: fixed; bottom: 22px; right: 22px; z-index: 9999;
    display: flex; flex-direction: column; align-items: flex-end; gap: 8px;
}
#va-toggle {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border: none; border-radius: 50px; padding: 10px 18px;
    color: white; font-size: 0.82rem; font-weight: 700; cursor: pointer;
    box-shadow: 0 4px 18px rgba(99,102,241,0.55);
    display: flex; align-items: center; gap: 7px;
    transition: all 0.2s;
}
#va-toggle:hover { transform: scale(1.05); }
#va-bubble {
    background: rgba(15,23,42,0.97);
    border: 1px solid #6366f1;
    border-radius: 14px; padding: 10px 14px;
    font-size: 0.76rem; color: #a5b4fc;
    max-width: 240px; text-align: right;
    display: none; line-height: 1.5;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
}
#va-bubble.show { display: block; }
#va-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #10b981; display: inline-block;
    animation: pulse-dot 1.5s infinite;
}
@keyframes pulse-dot {
    0%,100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.4; transform: scale(0.6); }
}
</style>

<div id="va-widget">
    <div id="va-bubble" class="">
        🎙️ Voice Assistant Active<br>
        <span style="color:#64748b;font-size:0.7rem">Listening for alerts &amp; hazards</span>
    </div>
    <button id="va-toggle" onclick="vaToggle()">
        <span id="va-dot"></span>
        <span id="va-label">🎙️ Voice ON</span>
    </button>
</div>

<script>
window._vaEnabled = true;
window._vaQueue   = [];

function vaToggle() {
    window._vaEnabled = !window._vaEnabled;
    var lbl = document.getElementById('va-label');
    var dot = document.getElementById('va-dot');
    var bub = document.getElementById('va-bubble');
    if (window._vaEnabled) {
        lbl.textContent = '🎙️ Voice ON';
        dot.style.background = '#10b981';
        bub.classList.add('show');
        setTimeout(function(){ bub.classList.remove('show'); }, 2500);
        vaSpeak('Voice assistant enabled');
    } else {
        lbl.textContent = '🔇 Voice OFF';
        dot.style.background = '#ef4444';
        bub.classList.remove('show');
        if(window.speechSynthesis) window.speechSynthesis.cancel();
    }
}

function vaSpeak(text, rate, pitch) {
    if (!window._vaEnabled) return;
    if (!window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    var u = new SpeechSynthesisUtterance(text);
    u.rate  = rate  || 0.92;
    u.pitch = pitch || 1.05;
    u.volume = 0.95;
    u.lang = 'en-IN';
    window.speechSynthesis.speak(u);
    // Show bubble
    var bub = document.getElementById('va-bubble');
    if (bub) {
        bub.innerHTML = '🎙️ ' + text;
        bub.classList.add('show');
        setTimeout(function(){ bub.classList.remove('show'); }, 3500);
    }
}

function vaBeep(type) {
    try {
        var a = new (window.AudioContext || window.webkitAudioContext)();
        var configs = {
            alert:    [{f:880,t:0},{f:660,t:0.18},{f:880,t:0.36},{f:660,t:0.54}],
            pothole:  [{f:280,t:0,d:0.25,w:'sawtooth'}],
            hospital: [{f:523,t:0,d:0.3,w:'sine'},{f:659,t:0.2,d:0.3,w:'sine'},{f:784,t:0.4,d:0.3,w:'sine'}],
            nav:      [{f:700,t:0,d:0.1},{f:900,t:0.12,d:0.1}],
        };
        var steps = configs[type] || configs.alert;
        steps.forEach(function(s) {
            var o = a.createOscillator();
            var g = a.createGain();
            o.type = s.w || 'square';
            o.frequency.value = s.f;
            var d = s.d || 0.15;
            g.gain.setValueAtTime(0.6, a.currentTime + s.t);
            g.gain.exponentialRampToValueAtTime(0.001, a.currentTime + s.t + d);
            o.connect(g); g.connect(a.destination);
            o.start(a.currentTime + s.t);
            o.stop(a.currentTime + s.t + d + 0.05);
        });
    } catch(e) {}
}

// Expose globally for page scripts
window.vaSpeak = vaSpeak;
window.vaBeep  = vaBeep;

// Welcome message on load
window.addEventListener('load', function() {
    setTimeout(function() {
        vaBeep('nav');
        vaSpeak('VigiRoad voice assistant ready', 0.9);
    }, 1200);
});
</script>
"""


def speak_html(text: str, beep: str = None, delay_ms: int = 300) -> str:
    """
    Returns inline HTML that auto-triggers speech + optional beep when rendered.
    Use st.markdown(speak_html(...), unsafe_allow_html=True).
    """
    beep_js = f"vaBeep('{beep}');" if beep else ""
    safe    = text.replace("'", "\\'").replace('"', '\\"')
    return (
        f"<script>"
        f"setTimeout(function(){{"
        f"  {beep_js}"
        f"  if(window.vaSpeak) window.vaSpeak('{safe}');"
        f"}}, {delay_ms});"
        f"</script>"
    )


def nav_instruction_html(text: str, icon: str = "🗺️",
                          color: str = "#6366f1") -> str:
    """Render a navigation-style instruction card + auto-speak it."""
    safe = text.replace("'", "\\'")
    return (
        f"<div style='background:rgba(99,102,241,0.12);border:2px solid {color};"
        f"border-radius:12px;padding:12px 16px;margin:6px 0;"
        f"display:flex;align-items:center;gap:10px;'>"
        f"<span style='font-size:1.6rem'>{icon}</span>"
        f"<div style='font-size:0.9rem;font-weight:600;color:#c7d2fe;line-height:1.4'>{text}</div>"
        f"</div>"
        f"<script>if(window.vaSpeak)window.vaSpeak('{safe}');</script>"
    )


def pothole_warning_html(location: str, distance_m: int,
                          severity: str = "medium") -> str:
    """Pothole warning card + beep + voice."""
    colors = {"low": "#10b981", "medium": "#f59e0b", "severe": "#ef4444",
              "Moderate": "#f59e0b", "High": "#ef4444", "Critical": "#ef4444"}
    c      = colors.get(severity, "#f59e0b")
    text   = f"Warning! {severity} pothole detected in {distance_m} metres near {location}. Reduce speed."
    safe   = text.replace("'", "\\'")
    return (
        f"<div style='background:{c}11;border:2px solid {c};"
        f"border-radius:12px;padding:11px 15px;margin:5px 0;"
        f"display:flex;align-items:center;gap:10px;'>"
        f"<span style='font-size:1.5rem'>🕳️</span>"
        f"<div>"
        f"<div style='font-size:0.88rem;font-weight:700;color:{c}'>Pothole ahead — {distance_m}m</div>"
        f"<div style='font-size:0.76rem;color:#94a3b8'>{location} · {severity} severity</div>"
        f"</div></div>"
        f"<script>if(window.vaBeep)window.vaBeep('pothole');"
        f"setTimeout(function(){{if(window.vaSpeak)window.vaSpeak('{safe}');}},400);</script>"
    )


def hospital_proximity_html(name: str, distance_m: int, beds: int = 0,
                             phone: str = "") -> str:
    """Hospital proximity alert card + chime + voice."""
    beds_txt = f" · {beds} beds available" if beds else ""
    text     = (f"Hospital nearby. {name} is {distance_m} metres ahead"
                f"{beds_txt}. Contact {phone}." if phone else
                f"Hospital nearby. {name} is {distance_m} metres ahead{beds_txt}.")
    safe     = text.replace("'", "\\'")
    return (
        f"<div style='background:rgba(16,185,129,0.1);border:2px solid #10b981;"
        f"border-radius:12px;padding:11px 15px;margin:5px 0;"
        f"display:flex;align-items:center;gap:10px;'>"
        f"<span style='font-size:1.5rem'>🏥</span>"
        f"<div>"
        f"<div style='font-size:0.88rem;font-weight:700;color:#10b981'>"
        f"{name} — {distance_m}m ahead</div>"
        f"<div style='font-size:0.76rem;color:#94a3b8'>"
        f"{'📞 ' + phone if phone else ''}{beds_txt}</div>"
        f"</div></div>"
        f"<script>if(window.vaBeep)window.vaBeep('hospital');"
        f"setTimeout(function(){{if(window.vaSpeak)window.vaSpeak('{safe}');}},500);</script>"
    )


def police_proximity_html(name: str, distance_m: int, officer: str = "",
                           contact: str = "") -> str:
    """Police station proximity card + voice."""
    text = (f"Police station nearby. {name} is {distance_m} metres away."
            f" Officer {officer}." if officer else
            f"Police station {name} is {distance_m} metres away.")
    safe = text.replace("'", "\\'")
    return (
        f"<div style='background:rgba(59,130,246,0.1);border:2px solid #3b82f6;"
        f"border-radius:12px;padding:11px 15px;margin:5px 0;"
        f"display:flex;align-items:center;gap:10px;'>"
        f"<span style='font-size:1.5rem'>🚔</span>"
        f"<div>"
        f"<div style='font-size:0.88rem;font-weight:700;color:#60a5fa'>"
        f"{name} — {distance_m}m</div>"
        f"<div style='font-size:0.76rem;color:#94a3b8'>"
        f"{'👮 ' + officer + ' · ' if officer else ''}{'📞 ' + contact if contact else ''}</div>"
        f"</div></div>"
        f"<script>if(window.vaSpeak)window.vaSpeak('{safe}');</script>"
    )
