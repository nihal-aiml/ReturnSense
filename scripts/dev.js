/**
 * ReturnSense — Concurrent Dev Launcher
 * =======================================
 * Starts both the frontend dev server and FastAPI backend simultaneously.
 * Works on Windows (PowerShell) and Unix systems.
 *
 * Usage: node scripts/dev.js
 */

const { spawn } = require("child_process");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");

console.log("");
console.log("  ╔══════════════════════════════════════════════╗");
console.log("  ║        📦 ReturnSense Dev Environment        ║");
console.log("  ╠══════════════════════════════════════════════╣");
console.log("  ║  Frontend:  http://localhost:5173             ║");
console.log("  ║  API:       http://localhost:8000             ║");
console.log("  ║  API Docs:  http://localhost:8000/docs        ║");
console.log("  ╚══════════════════════════════════════════════╝");
console.log("");

// Start frontend server
const frontend = spawn("node", ["server.js"], {
  cwd: ROOT,
  stdio: "pipe",
  shell: true,
});

frontend.stdout.on("data", (data) => {
  const lines = data.toString().trim().split("\n");
  lines.forEach((line) => {
    if (line.trim()) console.log(`  [Frontend] ${line}`);
  });
});

frontend.stderr.on("data", (data) => {
  console.error(`  [Frontend ERROR] ${data.toString().trim()}`);
});

// Start FastAPI backend
const api = spawn("uvicorn", ["api.app:app", "--reload", "--port", "8000", "--host", "0.0.0.0"], {
  cwd: ROOT,
  stdio: "pipe",
  shell: true,
});

api.stdout.on("data", (data) => {
  const lines = data.toString().trim().split("\n");
  lines.forEach((line) => {
    if (line.trim()) console.log(`  [API] ${line}`);
  });
});

api.stderr.on("data", (data) => {
  const lines = data.toString().trim().split("\n");
  lines.forEach((line) => {
    if (line.trim()) console.log(`  [API] ${line}`);
  });
});

// Handle cleanup
function cleanup() {
  console.log("\n  Shutting down ReturnSense...");
  frontend.kill();
  api.kill();
  process.exit(0);
}

process.on("SIGINT", cleanup);
process.on("SIGTERM", cleanup);
process.on("exit", cleanup);

frontend.on("exit", (code) => {
  if (code !== null && code !== 0) {
    console.error(`  [Frontend] Process exited with code ${code}`);
  }
});

api.on("exit", (code) => {
  if (code !== null && code !== 0) {
    console.error(`  [API] Process exited with code ${code}`);
  }
});
