from flask import Flask, jsonify, send_from_directory
import examples
import os

app = Flask(__name__, static_url_path='')

@app.route('/')
def root():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def send_file(path):
    return send_from_directory('.', path)

@app.route('/run_example/<method>')
def run_example(method):
    try:
        if method == 'apriori':
            rules = examples.run_apriori_example()
            return jsonify({
                'text': str(rules.head()),
                'image': None
            })
        
        elif method == 'montecarlo':
            pi = examples.estimate_pi()
            return jsonify({
                'text': f'Estimated value of π: {pi:.6f}',
                'image': '/monte_carlo.png'
            })
        
        elif method == 'kmeans':
            centers = examples.run_kmeans_example()
            return jsonify({
                'text': f'Cluster centers:\n{centers}',
                'image': '/kmeans.png'
            })
        
        elif method == 'decision-trees':
            depth = examples.run_decision_tree_example()
            return jsonify({
                'text': f'Decision tree depth: {depth}',
                'image': '/decision_tree.png'
            })
        
        else:
            return jsonify({'error': 'Invalid method'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=8000)
