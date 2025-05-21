from flask import Flask, request, jsonify
import os
import cv2
import tempfile
from app import main as matcher_main, CONFIG  # assuming your logic is in app.py

app = Flask(__name__)

@app.route('/match', methods=['POST'])
def match_furniture():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    image = request.files['image']
    category = request.form.get('category', '')

    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp:
        image.save(temp.name)
        input_path = temp.name

    # Call main matcher logic
    CONFIG['input_path'] = input_path
    CONFIG['target_category'] = category

    try:
        matcher_main()
        return jsonify({"message": "Matching completed", "report": CONFIG['report_path'] + "/report.html"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(input_path)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render provides $PORT
    app.run(debug=False, host='0.0.0.0', port=port)

