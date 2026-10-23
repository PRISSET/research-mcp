import { IncomingMessage, ServerResponse } from 'http';
import { resolveToken } from '../services/token/auth-token-pool.service';

export function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on('data', (c: Buffer) => chunks.push(c));
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

/**
 * Raw bearer value exactly as sent by the client (or the env fallback). This
 * may be a single token or a pipe-delimited list: `token1|token2|token3`.
 */
export function getRawToken(req: IncomingMessage): string {
  const auth = req.headers.authorization;
  if (auth && auth.startsWith('Bearer ')) return auth.slice(7).trim();
  return process.env.DEEPSEEK_CHAT_TOKEN || '';
}

/**
 * Resolve ONE usable token for this request by rotating over the
 * pipe-delimited list and skipping tokens currently in rate-limit cooldown.
 */
export function getToken(req: IncomingMessage): string {
  return resolveToken(getRawToken(req));
}

export function sendJsonResponse(res: ServerResponse, statusCode: number, payload: unknown): void {
  if (res.headersSent) return;
  res.writeHead(statusCode, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(payload));
}

export function sendJsonError(res: ServerResponse, statusCode: number, message: string, type: string = 'api_error'): void {
  sendJsonResponse(res, statusCode, {
    error: { message, type },
  });
}
