import { IncomingMessage, ServerResponse } from 'http';
import { config } from '../config';
import { handleGetModels, handleListModels } from '../controllers/models.controller';
import { handleChatCompletions } from '../controllers/chat.controller';
import { handleResponses } from '../controllers/responses.controller';
import { getToken, getRawToken, sendJsonResponse, sendJsonError } from '../utils/http.utils';
import { getPoolInfo } from '../services/token/auth-token-pool.service';

export async function handleRoutes(req: IncomingMessage, res: ServerResponse): Promise<void> {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (req.method === 'OPTIONS') {
    res.writeHead(200);
    res.end();
    return;
  }

  const userToken = getToken(req);
  const url = new URL(req.url || '/', `http://${req.headers.host || '127.0.0.1'}`);
  const p = url.pathname.replace(/\/+$/, '') || '/';

  if (url.pathname === '/health' || url.pathname === '/status') {
    sendJsonResponse(res, 200, {
      status: 'ok',
      token_present: !!userToken,
      port: config.port,
      // When the bearer is a pipe-list (token1|token2|token3), this reports how
      // many tokens the pool holds and how many are currently cooling down
      // after a throttle response.
      token_pool: getPoolInfo(getRawToken(req)),
    });
    return;
  }

  if (p === '/v1/models' || p === '/models') {
    handleGetModels(req, res);
    return;
  }

  if (!userToken) {
    sendJsonError(res, 401, 'Missing DEEPSEEK_CHAT_TOKEN', 'auth_error');
    return;
  }

  try {
    if (p === '/api/v1/models' && req.method === 'GET') {
      await handleListModels(req, res, userToken);
    } else if (
      (p === '/v1/chat/completions' ||
        p === '/chat/completions' ||
        p === '/v1/completions' ||
        p === '/completions') &&
      req.method === 'POST'
    ) {
      await handleChatCompletions(req, res, userToken);
    } else if ((p === '/v1/responses' || p === '/responses') && req.method === 'POST') {
      await handleResponses(req, res, userToken);
    } else {
      sendJsonError(res, 404, `Not found: ${req.method} ${url.pathname}`, 'not_found');
    }
  } catch (err: any) {
    console.error('[proxy] Server error:', err);
    if (!res.headersSent) {
      sendJsonError(res, 500, err.message, 'internal_error');
    }
  }
}
