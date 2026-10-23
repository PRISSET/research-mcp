import { ClientRequest, IncomingMessage, ServerResponse } from 'http';
import https from 'https';
import crypto from 'crypto';
import { config } from '../config';
import { ChatCompletionRequest, ChatCompletionResponse, TokenUsage } from '../types/openai.types';
import { getPowResponseHeader } from '../services/pow/pow.service';
import { getOrCreateChatSession } from '../services/deepseek/session.service';
import { extractImagesFromMessages, uploadAndPrepareImage } from '../services/deepseek/vision.service';
import { formatMessagesToPrompt } from '../services/parser/prompt-formatter.service';
import { extractToolCalls } from '../services/parser/tool-extractor.service';
import { findStreamBoundary } from '../services/parser/stop-sequence.service';
import {
  createParserState,
  processDeepSeekSSEData,
  flushFenceBuffers,
} from '../services/parser/sse-parser.service';
import { estimatePromptTokens, estimateTextTokens } from '../services/token/token-counter.service';
import { readBody, sendJsonError, sendJsonResponse, getRawToken } from '../utils/http.utils';
import {
  reportRateLimited,
  reportSuccess,
  resolveTokenExcluding,
} from '../services/token/auth-token-pool.service';

interface UpstreamSuccess {
  ok: true;
  res: IncomingMessage;
  req: ClientRequest;
}

interface UpstreamFailure {
  ok: false;
  status: number;
  body: string;
}

/**
 * Statuses that mean "this token is throttled" rather than "this request is
 * malformed". They are the trigger for rotating to a sibling token.
 */
function isThrottleStatus(status: number): boolean {
  return status === 429 || status === 503 || status === 529;
}

/** Best-effort extraction of a `retry_after` hint from an upstream error body. */
function extractRetryAfterMs(body: string): number | undefined {
  try {
    const json = JSON.parse(body);
    const raw =
      json?.retry_after ??
      json?.data?.retry_after ??
      json?.error?.retry_after ??
      json?.data?.biz_data?.retry_after;
    if (typeof raw === 'number' && raw > 0) return raw > 1000 ? raw : raw * 1000;
    if (typeof raw === 'string') {
      const n = parseFloat(raw);
      if (!Number.isNaN(n) && n > 0) return n > 1000 ? n : n * 1000;
    }
  } catch (e) {}
  return undefined;
}

/**
 * Open the DeepSeek completion request and resolve as soon as we know whether
 * the token was accepted. A non-200 status is turned into a resolved failure so
 * the caller can decide to rotate instead of streaming an error to the client.
 */
function openUpstreamCompletion(
  userToken: string,
  payload: string,
  powHeader: string
): Promise<UpstreamSuccess | UpstreamFailure> {
  return new Promise((resolve) => {
    const dsReq = https.request(
      {
        hostname: config.targetHost,
        port: 443,
        path: '/api/v0/chat/completion',
        method: 'POST',
        headers: {
          Authorization: `Bearer ${userToken}`,
          'Content-Type': 'application/json',
          Accept: 'text/event-stream',
          'x-ds-pow-response': powHeader,
          ...config.clientHeaders,
        },
        timeout: config.timeoutMs,
      },
      (dsRes) => {
        if (dsRes.statusCode !== 200) {
          let errBody = '';
          dsRes.on('data', (c) => (errBody += c));
          dsRes.on('end', () =>
            resolve({ ok: false, status: dsRes.statusCode || 502, body: errBody })
          );
          return;
        }
        resolve({ ok: true, res: dsRes, req: dsReq });
      }
    );
    // A transport-level error is reported as status 0 so it is never mistaken
    // for a throttle and never triggers rotation.
    dsReq.on('error', (err) => resolve({ ok: false, status: 0, body: err.message }));
    dsReq.write(payload);
    dsReq.end();
  });
}

export async function handleChatCompletions(req: IncomingMessage, res: ServerResponse, userToken: string): Promise<void> {
  const bodyText = await readBody(req);
  let parsed: ChatCompletionRequest;

  try {
    parsed = JSON.parse(bodyText);
  } catch (e) {
    sendJsonError(res, 400, 'Invalid JSON', 'invalid_request_error');
    return;
  }

  const messages = parsed.messages || [];
  const model = parsed.model || 'deepseek-chat';
  let stream = parsed.stream !== false;
  if (config.forceStream) {
    stream = true;
  }
  const tools = parsed.tools || parsed.functions || [];

  const imageSources = extractImagesFromMessages(messages);

  const refFileIds: string[] = [];
  if (imageSources.length > 0) {
    for (const src of imageSources) {
      try {
        const fid = await uploadAndPrepareImage(userToken, src);
        if (fid) refFileIds.push(fid);
      } catch (err: any) {
        console.error('[proxy] Error uploading image:', err.message);
      }
    }
  }

  const modelType = 'default';

  let thinkingEnabled = false;
  if (parsed.thinking_enabled !== undefined) {
    thinkingEnabled = parsed.thinking_enabled;
  } else if (parsed.reasoning !== undefined) {
    thinkingEnabled = parsed.reasoning;
  }

  const searchEnabled = true;

  // Распаковываем структурированный промпт с tools
  const prompt = formatMessagesToPrompt(messages, tools, parsed.client_prompt === true);
  const completionId = `chatcmpl-${crypto.randomUUID()}`;
  const created = Math.floor(Date.now() / 1000);

  // The raw bearer value may be a pipe-delimited token list
  // (`token1|token2|token3`). `userToken` is the token already selected for this
  // request; if it turns out to be throttled we rotate to a sibling token and
  // replay the whole upstream exchange with it.
  const rawToken = getRawToken(req);
  const maxRetries = Math.max(0, config.authToken.maxRetries);
  // Rotating mid-request would orphan uploaded ref_file_ids (they belong to the
  // account that uploaded them), so a request that carries images stays pinned
  // to the token it started with.
  const canRotate = refFileIds.length === 0;
  let currentToken = userToken;
  let attempt = 0;

  const rotateOrGiveUp = (): boolean => {
    if (!canRotate || attempt > maxRetries) return false;
    const next = resolveTokenExcluding(rawToken, currentToken);
    if (!next || next === currentToken) return false;
    currentToken = next;
    return true;
  };

  /**
   * Hand an accepted upstream response to the client. Runs the original
   * SSE/non-SSE translation; by the time this is called the token question is
   * settled (status 200 already received), so it never rotates.
   */
  const pumpUpstream = (dsRes: IncomingMessage, dsReq: ClientRequest): void => {
    let fullContent = '';
    let finalUsage: TokenUsage | null = null;
    const hasTools = tools && tools.length > 0;
    // OpenAI-compatible clients signal via `stream_options.include_usage`
    // that they want a dedicated final chunk carrying `usage` with an empty
    // `choices` array. Operators can force it for every client with
    // STREAM_USAGE_ALWAYS=true.
    const includeUsageChunk =
      (parsed as any).stream_options?.include_usage === true ||
      config.usage.alwaysIncludeStreamUsage;
    let firstChunkSent = false;
    let finalized = false;

    const state = createParserState();

    // Token accounting.
    //
    // Prompt side: DeepSeek's SSE never reports an input token count, so the
    // number is derived from the exact formatted prompt that was sent
    // upstream (system scaffold + serialized tool schemas included). That is
    // the ground truth for pressure against the 64k context window.
    //
    // Completion side: DeepSeek's measured `accumulated_token_usage` is
    // preferred; the estimator is only a fallback for streams that ended
    // without one. `total_tokens` is always the arithmetic sum, so harnesses
    // that cross-check the three fields never see an inconsistency.
    const estimatedPromptTokens = estimatePromptTokens(prompt);
    const computeUsage = (): TokenUsage => {
      const accumulated = finalUsage?.completion_tokens;
      let promptTokens = estimatedPromptTokens;
      let completionTokens: number;

      if (typeof accumulated === 'number' && accumulated > 0) {
        if (config.usage.accumulatedIsTotal) {
          promptTokens = Math.min(estimatedPromptTokens, accumulated);
          completionTokens = accumulated - promptTokens;
        } else {
          completionTokens = accumulated;
        }
      } else {
        completionTokens = estimateTextTokens(
          state.responseBuffer + state.reasoningBuffer
        );
      }

      const totalTokens = promptTokens + completionTokens;
      return {
        inputTokens: promptTokens,
        outputTokens: completionTokens,
        totalTokens: totalTokens,
        input: promptTokens,
        output: completionTokens,
        prompt_tokens: promptTokens,
        completion_tokens: completionTokens,
        total_tokens: totalTokens,
      };
    };

    const abortUpstream = () => {
      try {
        dsRes.destroy();
      } catch (e) {}
      try {
        dsReq.destroy();
      } catch (e) {}
    };

    // The hardcoded stop sequence was found. The phrase itself has already
    // been excluded from the emitted deltas, so here we only cut the upstream
    // stream immediately and close the client stream with "stop".
    // Объявляем заранее: обработчик нормального завершения upstream нужен
    // ещё и при срабатывании стоп-последовательности.
    let onUpstreamEnd: () => void;

    const finalizeWithStop = () => {
      if (finalized) return;
      // Стоп-последовательность: отдаём всё накопленное (tool calls и текст
      // до стоп-фразы) точно так же, как при нормальном завершении потока,
      // и только затем рвём upstream.
      onUpstreamEnd();
      abortUpstream();
    };

    const flushBuffers = (isFinal = false) => {
      if (!stream) return;
      if (finalized) return;

      // 1. Инициализирующий чанк с ролью
      if (!firstChunkSent) {
        firstChunkSent = true;
        const initChunk = {
          id: completionId,
          object: 'chat.completion.chunk',
          created: created,
          model: model,
          choices: [
            {
              index: 0,
              delta: { role: 'assistant' },
              finish_reason: null,
            },
          ],
        };
        res.write(`data: ${JSON.stringify(initChunk)}\n\n`);
      }

      // 2/3. Останавливаемся на хардкодных последовательностях: стоп-фразе и
      // фразе замены. Проверяем и reasoning, и видимый ответ; хвост, который
      // может быть началом любой из фраз, придерживаем, чтобы она не утекла
      // частично. Фраза замены не отбрасывается, а подменяется на REPLACE_WITH.
      const rBoundary = findStreamBoundary(
        state.reasoningBuffer,
        state.lastReasoningSent,
        isFinal
      );
      const cBoundary = findStreamBoundary(
        state.responseBuffer,
        state.lastResponseSent,
        isFinal
      );
      const stopped = rBoundary.stopped || cBoundary.stopped;
      const replaced = rBoundary.replaced || cBoundary.replaced;
      const terminal = stopped || replaced;
      const rTerminal = rBoundary.stopped || rBoundary.replaced;
      const cTerminal = cBoundary.stopped || cBoundary.replaced;

      // Если срабатывание произошло в одном буфере, поток всё равно
      // завершается, поэтому второй буфер отдаём целиком, ничего не придерживая.
      const rEnd = terminal && !rTerminal ? state.reasoningBuffer.length : rBoundary.emitEnd;
      const cEnd = terminal && !cTerminal ? state.responseBuffer.length : cBoundary.emitEnd;

      // 2. Стримим Reasoning ВСЕГДА (даже если есть tools)
      if (rEnd > state.lastReasoningSent || (rBoundary.replaced && rBoundary.replacement)) {
        const newReasoning =
          state.reasoningBuffer.slice(state.lastReasoningSent, rEnd) +
          (rBoundary.replaced ? rBoundary.replacement : '');
        state.lastReasoningSent = rEnd;

        const reasoningChunk = {
          id: completionId,
          object: 'chat.completion.chunk',
          created: created,
          model: model,
          choices: [
            {
              index: 0,
              delta: {
                reasoning_content: newReasoning,
                reasoning: newReasoning,
              },
              finish_reason: null,
            },
          ],
        };
        res.write(`data: ${JSON.stringify(reasoningChunk)}\n\n`);
      }

      // 3. Текст ответа стримим только если нет tools
      if (
        !hasTools &&
        (cEnd > state.lastResponseSent || (cBoundary.replaced && cBoundary.replacement))
      ) {
        const newResponse =
          state.responseBuffer.slice(state.lastResponseSent, cEnd) +
          (cBoundary.replaced ? cBoundary.replacement : '');
        state.lastResponseSent = cEnd;
        fullContent += newResponse;

        const contentChunk = {
          id: completionId,
          object: 'chat.completion.chunk',
          created: created,
          model: model,
          choices: [
            {
              index: 0,
              delta: { content: newResponse },
              finish_reason: null,
            },
          ],
        };
        res.write(`data: ${JSON.stringify(contentChunk)}\n\n`);
      }

      if (terminal) {
        // Синхронизируем буферы с тем, что уже отдано: фраза замены
        // подменяется на REPLACE_WITH, стоп-фраза просто отбрасывается.
        state.reasoningBuffer =
          state.reasoningBuffer.slice(0, rEnd) +
          (rBoundary.replaced ? rBoundary.replacement : '');
        state.responseBuffer =
          state.responseBuffer.slice(0, cEnd) +
          (cBoundary.replaced ? cBoundary.replacement : '');
        state.lastReasoningSent = state.reasoningBuffer.length;
        state.lastResponseSent = state.responseBuffer.length;
        finalizeWithStop();
      }
    };

    if (stream) {
      res.writeHead(200, {
        'Content-Type': 'text/event-stream; charset=utf-8',
        'Cache-Control': 'no-cache',
        Connection: 'keep-alive',
        'Access-Control-Allow-Origin': '*',
      });

      let buffer = '';
      dsRes.on('data', (chunk) => {
        if (finalized) return;
        buffer += chunk.toString('utf8');
        let events = buffer.split('\n\n');
        buffer = events.pop() || '';

        for (const event of events) {
          if (finalized) break;
          const lines = event.split('\n');
          let eventName: string | null = null;
          let dataStr: string | null = null;

          for (const line of lines) {
            const trimmedLine = line.trim();
            if (trimmedLine.startsWith('event:')) {
              eventName = trimmedLine.slice(6).trim();
            } else if (trimmedLine.startsWith('data:')) {
              dataStr = trimmedLine.slice(5).trim();
            }
          }

          if (dataStr) {
            try {
              const data = JSON.parse(dataStr);
              processDeepSeekSSEData(data, state, eventName);
              flushBuffers();
              if (state.finalUsage) finalUsage = state.finalUsage;
            } catch (e) {}
          }
        }
      });

      onUpstreamEnd = () => {
        if (finalized) return;

        // Сначала выпускаем хвост, придержанный нормализатором fence'ов:
        // иначе последние символы экранированного блока не попадут в буферы
        // и tool call будет извлечён неполным. Вызов идемпотентен.
        flushFenceBuffers(state);

        // Отдаём придержанный хвост; если стоп-фраза завершилась ровно
        // здесь, поток финализируется с finish_reason "stop".
        flushBuffers(true);
        if (finalized) return;

        const usage = computeUsage();

        if (hasTools) {
          const { toolCalls, content: cleanText } = extractToolCalls(
            state.responseBuffer,
            tools,
            state.reasoningBuffer
          );

          if (cleanText) {
            const textChunk = {
              id: completionId,
              object: 'chat.completion.chunk',
              created: created,
              model: model,
              choices: [
                {
                  index: 0,
                  delta: { content: cleanText },
                  finish_reason: null,
                },
              ],
            };
            res.write(`data: ${JSON.stringify(textChunk)}\n\n`);
          }

          if (toolCalls.length > 0) {
            for (let i = 0; i < toolCalls.length; i++) {
              const tc = toolCalls[i];
              const callChunk = {
                id: completionId,
                object: 'chat.completion.chunk',
                created: created,
                model: model,
                choices: [
                  {
                    index: 0,
                    delta: {
                      tool_calls: [
                        {
                          index: i,
                          id: tc.id,
                          type: 'function',
                          function: {
                            name: tc.name,
                            arguments: tc.arguments,
                          },
                        },
                      ],
                    },
                    finish_reason: null,
                  },
                ],
              };
              res.write(`data: ${JSON.stringify(callChunk)}\n\n`);
            }
            res.write(
              `data: ${JSON.stringify({
                id: completionId,
                object: 'chat.completion.chunk',
                created,
                model,
                choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }],
                usage,
              })}\n\n`
            );
          } else {
            res.write(
              `data: ${JSON.stringify({
                id: completionId,
                object: 'chat.completion.chunk',
                created,
                model,
                choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
                usage,
              })}\n\n`
            );
          }
        } else {
          res.write(
            `data: ${JSON.stringify({
              id: completionId,
              object: 'chat.completion.chunk',
              created,
              model,
              choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
              usage,
            })}\n\n`
          );
        }

        if (includeUsageChunk) {
          res.write(
            `data: ${JSON.stringify({
              id: completionId,
              object: 'chat.completion.chunk',
              created,
              model,
              choices: [],
              usage,
            })}\n\n`
          );
        }

        res.write('data: [DONE]\n\n');
        finalized = true;
        res.end();
      };

      dsRes.on('end', onUpstreamEnd);

      dsRes.on('error', (err) => {
        if (finalized) return;
        finalized = true;
        console.error('[proxy] Stream error:', err);
        if (!res.headersSent) {
          sendJsonError(res, 502, err.message, 'api_error');
        } else {
          try {
            res.end();
          } catch (e) {}
        }
      });
    } else {
      let responseBody = '';
      dsRes.on('data', (c) => (responseBody += c));
      dsRes.on('end', () => {
        const events = responseBody.split('\n\n');
        for (const event of events) {
          const lines = event.split('\n');
          let eventName: string | null = null;
          let dataStr: string | null = null;

          for (const line of lines) {
            const trimmedLine = line.trim();
            if (trimmedLine.startsWith('event:')) {
              eventName = trimmedLine.slice(6).trim();
            } else if (trimmedLine.startsWith('data:')) {
              dataStr = trimmedLine.slice(5).trim();
            }
          }

          if (dataStr) {
            try {
              const data = JSON.parse(dataStr);
              processDeepSeekSSEData(data, state, eventName);
              if (state.finalUsage) finalUsage = state.finalUsage;
            } catch (e) {}
          }
        }

        // Сначала выпускаем хвост, придержанный нормализатором fence'ов,
        // чтобы границы считались по полному тексту.
        flushFenceBuffers(state);

        // Применяем хардкодные последовательности к накопленному ответу.
        // В non-stream режиме придерживать нечего, поэтому неполный префикс
        // считается обычным текстом: полная стоп-фраза обрезает вывод, а
        // полная фраза замены подменяется на REPLACE_WITH.
        const rBoundary = findStreamBoundary(state.reasoningBuffer, 0, true);
        const cBoundary = findStreamBoundary(state.responseBuffer, 0, true);
        state.reasoningBuffer =
          state.reasoningBuffer.slice(0, rBoundary.emitEnd) +
          (rBoundary.replaced ? rBoundary.replacement : '');
        state.responseBuffer =
          state.responseBuffer.slice(0, cBoundary.emitEnd) +
          (cBoundary.replaced ? cBoundary.replacement : '');

        // Same accounting as the streaming path: estimated prompt from the
        // formatted upstream prompt, measured completion when DeepSeek
        // reported it, estimator otherwise.
        const usage: TokenUsage = computeUsage();

        const { toolCalls, content: cleanText, cleanReasoning } = extractToolCalls(
          state.responseBuffer,
          tools,
          state.reasoningBuffer
        );

        const choice: any = {
          index: 0,
          message: {
            role: 'assistant',
            content: cleanText,
            ...(state.reasoningBuffer
              ? {
                  reasoning_content: cleanReasoning || state.reasoningBuffer,
                  reasoning: cleanReasoning || state.reasoningBuffer,
                }
              : {}),
          },
          finish_reason: toolCalls.length > 0 ? 'tool_calls' : 'stop',
        };

        if (toolCalls.length > 0) {
          choice.message.tool_calls = toolCalls.map((tc) => ({
            id: tc.id,
            type: 'function',
            function: { name: tc.name, arguments: tc.arguments },
          }));
        }

        const out: ChatCompletionResponse = {
          id: completionId,
          object: 'chat.completion',
          created: created,
          model: model,
          choices: [choice],
          usage: usage,
        };

        if (!res.headersSent) {
          sendJsonResponse(res, 200, out);
        }
      });

      dsRes.on('error', (err) => {
        console.error('[proxy] Non-stream error:', err);
        if (!res.headersSent) {
          sendJsonError(res, 502, err.message, 'api_error');
        }
      });
    }

    dsReq.on('error', (err) => {
      // Stop-sequence abort destroys this request on purpose; the client stream
      // is already closed, so don't log a misleading error for it.
      if (res.writableEnded) return;
      console.error('[proxy] Request error:', err);
      if (!res.headersSent) {
        sendJsonError(res, 502, err.message, 'api_error');
      }
    });
  };

  // ---------------------------------------------------------------------------
  // Request loop: try the selected token; on a throttle status rotate to a
  // sibling token and replay PoW + session + completion with it.
  // ---------------------------------------------------------------------------
  while (true) {
    attempt += 1;

    let powHeader = '';
    try {
      powHeader = await getPowResponseHeader(currentToken, '/api/v0/chat/completion');
    } catch (err: any) {
      if (isThrottleStatus(err.status) && rotateOrGiveUp()) {
        reportRateLimited(currentToken);
        console.warn(
          `[proxy] PoW throttled for a token; rotating to a sibling token (attempt ${attempt}).`
        );
        continue;
      }
      console.error('[proxy] PoW Error:', err.message);
      if (!res.headersSent) {
        sendJsonError(res, 502, `PoW error: ${err.message}`, 'api_error');
      }
      return;
    }

    let sessionId = parsed.chat_session_id;
    if (!sessionId) {
      try {
        sessionId = await getOrCreateChatSession(currentToken, 'chat');
      } catch (err: any) {
        if (isThrottleStatus(err.status) && rotateOrGiveUp()) {
          reportRateLimited(currentToken);
          console.warn(
            `[proxy] Session creation throttled for a token; rotating to a sibling token (attempt ${attempt}).`
          );
          continue;
        }
        console.error('[proxy] Session error:', err.message);
        if (!res.headersSent) {
          sendJsonError(res, 502, `Session error: ${err.message}`, 'api_error');
        }
        return;
      }
    }

    const dsPayload = JSON.stringify({
      chat_session_id: sessionId,
      parent_message_id: null,
      prompt: prompt,
      ref_file_ids: refFileIds,
      thinking_enabled: true,
      search_enabled: false,
      model_type: modelType,
      action: null,
      preempt: false,
    });

    const upstream = await openUpstreamCompletion(currentToken, dsPayload, powHeader);

    if (upstream.ok) {
      reportSuccess(currentToken);
      pumpUpstream(upstream.res, upstream.req);
      return;
    }

    if (isThrottleStatus(upstream.status) && rotateOrGiveUp()) {
      reportRateLimited(currentToken, extractRetryAfterMs(upstream.body));
      console.warn(
        `[proxy] Token throttled (status ${upstream.status}); rotating to a sibling token (attempt ${attempt}).`
      );
      continue;
    }

    if (upstream.status === 0) {
      console.error('[proxy] Request error:', upstream.body);
      if (!res.headersSent) {
        sendJsonError(res, 502, upstream.body, 'api_error');
      }
      return;
    }

    console.error(`[proxy] DeepSeek upstream error ${upstream.status}:`, upstream.body);
    if (!res.headersSent) {
      sendJsonError(res, upstream.status || 502, `DeepSeek Web Error: ${upstream.body}`, 'api_error');
    }
    return;
  }
}
