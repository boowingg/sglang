import { createReadStream } from "node:fs";
import { access, stat } from "node:fs/promises";
import http from "node:http";
import path from "node:path";
import { buildBook, repoRoot, webRoot } from "./build-content.mjs";

const MIME_TYPES = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".jpeg": "image/jpeg",
  ".jpg": "image/jpeg",
  ".md": "text/markdown; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".png": "image/png",
  ".py": "text/plain; charset=utf-8",
  ".c": "text/plain; charset=utf-8",
  ".cc": "text/plain; charset=utf-8",
  ".cpp": "text/plain; charset=utf-8",
  ".cu": "text/plain; charset=utf-8",
  ".cuh": "text/plain; charset=utf-8",
  ".h": "text/plain; charset=utf-8",
  ".hip": "text/plain; charset=utf-8",
  ".svg": "image/svg+xml",
};

function safeResolve(root, requestedPath) {
  const resolved = path.resolve(root, `.${requestedPath}`);
  return resolved === root || resolved.startsWith(`${root}${path.sep}`) ? resolved : null;
}

async function sendFile(request, response, filePath, cacheControl = "no-cache") {
  try {
    const fileStats = await stat(filePath);
    if (!fileStats.isFile()) return false;
    response.writeHead(200, {
      "Content-Type": MIME_TYPES[path.extname(filePath).toLowerCase()] || "application/octet-stream",
      "Content-Length": fileStats.size,
      "Cache-Control": cacheControl,
      "X-Content-Type-Options": "nosniff",
    });
    if (request.method === "HEAD") response.end();
    else createReadStream(filePath).pipe(response);
    return true;
  } catch {
    return false;
  }
}

await buildBook();

const server = http.createServer(async (request, response) => {
  try {
    const url = new URL(request.url || "/", "http://localhost");
    const decodedPath = decodeURIComponent(url.pathname);

    if (decodedPath.startsWith("/__repo/")) {
      const repositoryPath = `/${decodedPath.slice("/__repo/".length)}`;
      const filePath = safeResolve(repoRoot, repositoryPath);
      if (filePath && (await sendFile(request, response, filePath))) return;
      response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
      response.end("Source file not found");
      return;
    }

    const requested = decodedPath === "/" ? "/index.html" : decodedPath;
    const filePath = safeResolve(webRoot, requested);
    const immutable = requested.startsWith("/vendor/") || requested.startsWith("/assets/");
    if (
      filePath &&
      (await sendFile(
        request,
        response,
        filePath,
        immutable ? "public, max-age=31536000, immutable" : "no-cache",
      ))
    ) {
      return;
    }

    if (!path.extname(decodedPath) && (await sendFile(request, response, path.join(webRoot, "index.html")))) {
      return;
    }
    response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
    response.end("Not found");
  } catch (error) {
    response.writeHead(500, { "Content-Type": "text/plain; charset=utf-8" });
    response.end(error instanceof Error ? error.message : "Internal server error");
  }
});

const initialPort = Number(process.env.PORT || process.argv[2] || 4173);
let port = initialPort;

function listen() {
  server.once("error", (error) => {
    if (error.code === "EADDRINUSE" && port < initialPort + 20) {
      port += 1;
      listen();
      return;
    }
    throw error;
  });
  server.listen(port, "127.0.0.1", () => {
    console.log(`HiCache source guide: http://127.0.0.1:${port}`);
  });
}

await access(path.join(webRoot, "index.html"));
listen();
