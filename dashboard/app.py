from flask import Flask, request, render_template
from flask_socketio import SocketIO
import sqlite3, os, datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, async_mode='eventlet') 

UPLOAD_DIR = "static/captures"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def init_db():
    con = sqlite3.connect("events.db")
    # Must include label and confidence!
    con.execute("CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, timestamp TEXT, image TEXT, label TEXT, confidence TEXT)")
    con.commit(); con.close()

@app.route("/upload", methods=["POST"])
def upload():
    ts = request.form.get("timestamp", datetime.datetime.now().isoformat())
    # Grab the new AI data, default to Unknown/0 if it fails
    label = request.form.get("label", "Unknown")
    confidence = request.form.get("confidence", "0.00")
    
    img = request.files["image"]
    filename = f"capture_{ts.replace(':','-')}.jpg"
    img.save(os.path.join(UPLOAD_DIR, filename))
    
    con = sqlite3.connect("events.db")
    con.execute("INSERT INTO events (timestamp, image, label, confidence) VALUES (?, ?, ?, ?)", (ts, filename, label, confidence))
    con.commit(); con.close()
    
    # Broadcast the new event AND the AI data to web clients
    socketio.emit('new_event', {
        'timestamp': ts, 
        'image': filename,
        'label': label,
        'confidence': confidence
    })
    
    return "OK", 200

@app.route("/")
def dashboard():
    con = sqlite3.connect("events.db")
    events = con.execute("SELECT * FROM events ORDER BY id DESC LIMIT 50").fetchall()
    con.close()
    return render_template("dashboard.html", events=events)
    
@app.route("/trigger", methods=["POST"])
def trigger_manual():
    # This sends a 'remote_capture' signal to any connected Pi Zero
    socketio.emit('remote_capture', {'data': 'trigger'})
    return "Triggered", 200

if __name__ == "__main__":
    init_db()
    socketio.run(app, host="0.0.0.0", port=5000)
