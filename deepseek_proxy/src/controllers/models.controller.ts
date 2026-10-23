import { IncomingMessage, ServerResponse } from 'http';
import https from 'https';
import { sendJsonResponse, sendJsonError } from '../utils/http.utils';
import { config } from '../config';
import { reportRateLimited } from '../services/token/auth-token-pool.service';

export async function handleListModels(req: IncomingMessage, res: ServerResponse, userToken: string): Promise<void> {
  try {
    const options = {
      hostname: 'openrouter.ai',
      port: 443,
      path: '/api/v1/models',
      method: 'GET',
      headers: {
        Authorization: `Bearer ${userToken}`,
        'Content-Type': 'application/json',
      },
    };

    const proxyReq = https.request(options, (proxyRes) => {
      let data = '';
      proxyRes.on('data', (chunk) => (data += chunk));
      proxyRes.on('end', () => {
        const status = proxyRes.statusCode || 502;
        // OpenRouter is a separate credential space from DeepSeek, but a 429
        // here still means "this token is throttled" — record it so a pipe-list
        // bearer backs off this token instead of hammering it.
        if (status === 429) {
          reportRateLimited(userToken);
        }
        if (status !== 200) {
          sendJsonError(res, status, `OpenRouter error: ${data}`, 'api_error');
          return;
        }
        try {
          const parsed = JSON.parse(data);
          sendJsonResponse(res, 200, parsed);
        } catch (err) {
          sendJsonError(res, 502, 'Failed to parse OpenRouter response', 'bad_gateway');
        }
      });
    });

    proxyReq.on('error', (err) => {
      console.error('[proxy] Models list error:', err.message);
      sendJsonError(res, 502, err.message, 'bad_gateway');
    });

    proxyReq.end();
  } catch (err: any) {
    console.error('[proxy] handleListModels error:', err.message);
    sendJsonError(res, 500, err.message, 'internal_error');
  }
}

export function handleGetModels(req: IncomingMessage, res: ServerResponse): void {
  const model = config.models.default;
  sendJsonResponse(res, 200, {
    object: 'list',
    data: [
      {
        id: model.id,
        object: 'model',
        created: 1700000000,
        owned_by: 'deepseek',
        // Context-window metadata. Different harnesses read different field
        // names for the same number, so the common aliases are exposed and
        // kept in sync with config.models — this is what lets a client decide
        // when to compact or inject context against the real 64k limit.
        context_length: model.contextWindow,
        context_window: model.contextWindow,
        max_context_length: model.contextWindow,
        max_input_tokens: model.contextWindow,
        max_output_tokens: model.maxTokens,
        max_tokens: model.maxTokens,
        // Modality metadata. Harness's own discovery ignores these fields, but
        // OpenAI/OpenRouter-compatible clients read them to decide whether they
        // may send images, so they are kept in sync with the `input`
        // declaration written into settings.yaml by addProvider().
        architecture: {
          modality: 'text+image->text',
          input_modalities: ['text', 'image'],
          output_modalities: ['text'],
        },
        input_modalities: ['text', 'image'],
        output_modalities: ['text'],
        reasoning: {
          efforts: [
            { id: 'off', name: 'Off', description: 'No reasoning' },
            { id: 'low', name: 'Low', description: 'Light reasoning' },
            { id: 'medium', name: 'Medium', description: 'Balanced reasoning' },
            { id: 'high', name: 'High', description: 'Deep reasoning' },
          ],
          defaultEffort: 'low',
        },
      },
    ],
  });
}
