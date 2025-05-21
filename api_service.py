from flask import Flask, request, jsonify
import os
import cv2
import tempfile
from app import main as matcher_main  # Remove CONFIG import

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

    # Construct config dictionary
    config = {
        "csv_path": "./ikea.csv",
        "input_path": input_path,
        "cache_dir": "./image_cache_ssim",
        "report_path": "./report",
        "target_category": category,
        "max_workers": 16,
        "image_size": (300, 300),
        "request_timeout": 10,
        "request_headers": {'User-Agent': 'Mozilla/5.0'}
    }

    try:
        matcher_main(config)  # Pass the config here
        return jsonify({
            "message": "Matching completed",
            "report": os.path.join(config['report_path'], "report.html")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(input_path)

if __name__ == '__main__':
    app.run(debug=True, port=10000, host='0.0.0.0')  # Added port binding for Render
