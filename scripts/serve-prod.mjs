/**
 * AstraAtmos 生产静态服务器 + /api 反向代理 + gzip 压缩
 *
 * 用途：替代 `npm run dev`，走隧道时快得多。
 *   - dev 模式：浏览器要下载上千个模块（每个一次 HTTP 请求），走隧道极慢
 *   - 生产模式：只有 index.html + 1 个 JS + 1 个 CSS，且带缓存哈希
 *
 * 在 Vite 里这个配置属于 preview 的 allowedHosts 范畴，见 vite.config.js。
 * 零依赖：只用 Node 内置模块，不 spawn 子进程。
 */
import http from 'node:http';
import { createReadStream } from 'node:fs';
import { stat, readFile } from 'node:fs/promises';
import { extname, join, normalize, resolve, sep } from 'node:path';
import { gzipSync, brotliCompressSync, constants as zlibConstants } from 'node:zlib';

const ROOT = resolve(process.env.ROOT || 'frontend/dist');
const PORT = Number(process.env.PORT || 5173);
const HOST = process.env.HOST || '0.0.0.0';
const API_HOST = process.env.API_HOST || '127.0.0.1';
const API_PORT = Number(process.env.API_PORT || 8000);
const COMPRESS_MIN = 1024; // 小于 1KB 不压缩

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.gif': 'image/gif',
  '.ico': 'image/x-icon',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.ttf': 'font/ttf',
  '.map': 'application/json; charset=utf-8',
  '.txt': 'text/plain; charset=utf-8',
  '.wasm': 'application/wasm',
  '.task': 'application/octet-stream',
  '.bin': 'application/octet-stream',
};

// 可压缩的文本类型
const COMPRESSIBLE = new Set([
  '.html', '.js', '.mjs', '.css', '.json', '.svg', '.txt', '.map',
]);

// 压缩结果缓存：filePath+size+mtime -> { gzip, br }
// 必须带 mtime：重新构建后文件长度可能恰好不变（例如 index.html 两次构建同为 1649 字节），
// 只用 size 做键会把上一版的压缩结果发给客户端，导致 index.html 仍引用已删除的哈希资源。
const compressCache = new Map();

function getEncoding(req) {
  const accept = String(req.headers['accept-encoding'] || '');
  if (/\bbr\b/.test(accept)) return 'br';
  if (/\bgzip\b/.test(accept)) return 'gzip';
  return null;
}

async function buildCompressed(filePath, size, mtimeMs) {
  const key = `${filePath}:${size}:${mtimeMs}`;
  const hit = compressCache.get(key);
  if (hit) return hit;
  // 同一文件被重写后旧键不再命中，顺手清掉，避免缓存无限增长。
  const prefix = `${filePath}:`;
  for (const existing of compressCache.keys()) {
    if (existing.startsWith(prefix) && existing !== key) compressCache.delete(existing);
  }
  const raw = await readFile(filePath);
  const entry = {
    gzip: gzipSync(raw, { level: 6 }),
    br: brotliCompressSync(raw, {
      params: { [zlibConstants.BROTLI_PARAM_QUALITY]: 5 },
    }),
  };
  compressCache.set(key, entry);
  const saved = Math.round(100 * (1 - entry.br.length / raw.length));
  console.log(
    `[compress] ${filePath.split(/[/\\]/).pop()} ${(raw.length / 1048576).toFixed(2)}MB -> br ${(entry.br.length / 1048576).toFixed(2)}MB (省 ${saved}%)`
  );
  return entry;
}

function send(res, code, body, headers = {}) {
  res.writeHead(code, { 'Cache-Control': 'no-cache', ...headers });
  res.end(body);
}

async function serveFile(req, res, filePath, statusCode = 200) {
  const info = await stat(filePath).catch(() => null);
  if (!info || !info.isFile()) return false;

  const ext = extname(filePath).toLowerCase();
  const type = MIME[ext] || 'application/octet-stream';
  const isHashed = /-[A-Za-z0-9_-]{8,}\.[a-z0-9]+$/i.test(filePath);
  const headers = {
    'Content-Type': type,
    'Cache-Control': isHashed ? 'public, max-age=31536000, immutable' : 'no-cache',
    'Accept-Ranges': 'bytes',
    Vary: 'Accept-Encoding',
  };

  // 文本资源走压缩
  const enc = getEncoding(req);
  if (enc && COMPRESSIBLE.has(ext) && info.size >= COMPRESS_MIN) {
    try {
      const packed = await buildCompressed(filePath, info.size, info.mtimeMs);
      const buf = enc === 'br' ? packed.br : packed.gzip;
      headers['Content-Encoding'] = enc;
      headers['Content-Length'] = buf.length;
      if (req.method === 'HEAD') { res.writeHead(statusCode, headers); return res.end(), true; }
      res.writeHead(statusCode, headers);
      return res.end(buf), true;
    } catch {
      // 压缩失败则回退为明文
    }
  }

  headers['Content-Length'] = info.size;
  if (req.method === 'HEAD') { res.writeHead(statusCode, headers); return res.end(), true; }
  res.writeHead(statusCode, headers);
  createReadStream(filePath).pipe(res);
  return true;
}

function proxyApi(req, res) {
  const headers = { ...req.headers, host: `${API_HOST}:${API_PORT}` };
  delete headers.connection;
  const upstream = http.request(
    { host: API_HOST, port: API_PORT, method: req.method, path: req.url, headers },
    (up) => {
      res.writeHead(up.statusCode || 502, up.headers);
      up.pipe(res);
    }
  );
  upstream.on('error', (err) => {
    send(res, 502, `API proxy error: ${err.message}`, {
      'Content-Type': 'text/plain; charset=utf-8',
    });
  });
  req.pipe(upstream);
}

const server = http.createServer(async (req, res) => {
  try {
    const urlPath = decodeURIComponent((req.url || '/').split('?')[0]);

    if (urlPath === '/api' || urlPath.startsWith('/api/')) {
      return proxyApi(req, res);
    }

    const safe = normalize(urlPath).replace(/^([/\\])+/, '');
    const candidate = join(ROOT, safe || 'index.html');
    if (!candidate.startsWith(ROOT + sep) && candidate !== ROOT) {
      return send(res, 403, 'Forbidden');
    }
    if (await serveFile(req, res, candidate)) return;
    // SPA 回退
    if (await serveFile(req, res, join(ROOT, 'index.html'))) return;
    return send(res, 404, 'Not found');
  } catch (err) {
    return send(res, 500, `Server error: ${err.message}`, {
      'Content-Type': 'text/plain; charset=utf-8',
    });
  }
});

server.listen(PORT, HOST, () => {
  console.log(`[astraatmos] root   : ${ROOT}`);
  console.log(`[astraatmos] listen : http://${HOST}:${PORT}`);
  console.log(`[astraatmos] /api   -> http://${API_HOST}:${API_PORT}`);
  console.log(`[astraatmos] gzip/br: enabled (cached in memory)`);
});
