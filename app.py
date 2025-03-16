

from flask import Flask, Response, request, jsonify
from dl_demo5 import main2
# test
app = Flask(__name__)

dynamic_config = {}  # Store dynamic values

@app.route('/info', methods=['POST'])
def set_info():
    data = request.json
    required_keys = ["pdf_s3_path", "DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD", "DB_PORT"]

    if not all(key in data for key in required_keys):
        return jsonify({"error": "Missing required keys in the request."}), 400

    # Store the values dynamically
    dynamic_config.update(data)
    return jsonify({"message": "Info set successfully."})

@app.route('/run', methods=['GET'])
def trigger_script():
    try:
        if not dynamic_config:
            return Response("Error: Please provide info first via the /info endpoint.", status=400)

        output = main2(dynamic_config)  # Pass dynamic values to main2
        return Response(output, mimetype='text/plain')
    except Exception as e:
        return Response(f"Error: {str(e)}", status=500)

if __name__ == '__main__':
    app.run(debug=True, port=5000)



