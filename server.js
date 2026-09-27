/**
 * ReturnSense — Lightweight Dev Server
 * ======================================
 * Zero-dependency Node.js static file server for the frontend.
 * Serves files from the /frontend directory.
 *
 * Usage: node server.js
 * Default: http://localhost:5173
 */

const http = require("http");
const fs = require("fs");
const path = require("path");

const PORT = process.env.PORT || 5173;
const FRONTEND_DIR = path.join(__dirname, "frontend");

const MIME_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
  ".ttf": "font/ttf",
  ".eot": "application/vnd.ms-fontobject",
};

const server = http.createServer((req, res) => {
  // Parse URL, ignore query string
  let urlPath = req.url.split("?")[0];

  // Default to index.html
  if (urlPath === "/") urlPath = "/index.html";

  const filePath = path.join(FRONTEND_DIR, urlPath);
  const ext = path.extname(filePath).toLowerCase();

  // Security: prevent directory traversal
  if (!filePath.startsWith(FRONTEND_DIR)) {
    res.writeHead(403);
    res.end("Forbidden");
    return;
  }

  fs.readFile(filePath, (err, data) => {
    if (err) {
      if (err.code === "ENOENT") {
        // SPA fallback — serve index.html for unknown routes
        fs.readFile(path.join(FRONTEND_DIR, "index.html"), (err2, fallback) => {
          if (err2) {
            res.writeHead(404);
            res.end("Not Found");
          } else {
            res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
            res.end(fallback);
          }
        });
      } else {
        res.writeHead(500);
        res.end("Internal Server Error");
      }
      return;
    }

    const contentType = MIME_TYPES[ext] || "application/octet-stream";
    res.writeHead(200, {
      "Content-Type": contentType,
      "Cache-Control": "no-cache",
      "Access-Control-Allow-Origin": "*",
    });
    res.end(data);
  });
});

server.listen(PORT, () => {
  console.log("");
  console.log("  ╔══════════════════════════════════════════════╗");
  console.log("  ║                                              ║");
  console.log("  ║   📦 ReturnSense Frontend Dev Server         ║");
  console.log(`  ║   🌐 http://localhost:${PORT}                  ║`);
  console.log("  ║                                              ║");
  console.log("  ║   Serving: ./frontend/                       ║");
  console.log("  ║   Press Ctrl+C to stop                       ║");
  console.log("  ║                                              ║");
  console.log("  ╚══════════════════════════════════════════════╝");
  console.log("");
});
