import http.server
import socketserver
import subprocess

PORT = 5006

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain; charset=utf-8')
        self.end_headers()
        
        try:
            # Run the inference script and capture strict output
            result = subprocess.run(
                ["/Users/anubhavgupta/Desktop/Scaler1/.venv/bin/python", "inference.py"], 
                capture_output=True, 
                text=True
            )
            self.wfile.write(result.stdout.encode('utf-8'))
        except Exception as e:
            self.wfile.write(f"Error: {e}".encode('utf-8'))

if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Server started at http://localhost:{PORT}")
        httpd.serve_forever()
