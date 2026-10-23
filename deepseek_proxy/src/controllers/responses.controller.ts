import { IncomingMessage, ServerResponse } from 'http';
import { PassThrough } from 'stream';
import crypto from 'crypto';
import {
  ResponsesRequest,
  OpenAIMessage,
  OpenAIContentPart,
  OpenAITool,
} from '../types/openai.types';
import { handleChatCompletions } from './chat.controller';
import { readBody, sendJsonError, sendJsonResponse } from '../utils/http.utils';

/**
 * OpenAI Responses API (`POST /v1/responses`) front-end.
 *
 * The proxy's upstream is DeepSeek's web chat socket, whose request shape is
 * fixed by `handleChatCompletions`. Rather than fork that 760-line pipeline
 * (PoW solving, session reuse, token rotation, image upload, stop sequences,
 * tool extraction, fence unescaping, usage estimation), this controller
 * normalizes a Responses request into a chat-completions request, runs it
 * through the existing controller against a capturing pseudo-response, and
 * translates the resulting `chat.completion.chunk` stream into the
 * `response.*` event contract that pi-ai's Responses parser consumes.
 *
 * Only OpenAI-standard fields are read on the way in: `input` items of type
 * `message` / `function_call` / `function_call_output`, and content parts
 * `input_text`, `input_image` and `input_file` (base64 `file_data` or a data
 * URL). Images therefore arrive as bytes, never as a local filesystem path.
 */

/* ------------------------------------------------------------------ *
 * Request normalization
 * ------------------------------------------------------------------ */

function partIsImage(part: any): boolean {
  if (!part || typeof part !== 'object') return false;
  return (
    part.type === 'input_image' ||
    part.type === 'image_url' ||
    part.type === 'image' ||
    part.type === 'input_file'
  );
}

/** Collapse a part list to a plain string when it carries no binary content. */
function collapseParts(parts: OpenAIContentPart[]): string | OpenAIContentPart[] {
  if (parts.some(partIsImage)) return parts;
  return parts.map((p: any) => String(p?.text ?? '')).join('');
}

/**
 * Normalize a Responses `content` value into the chat-completions content
 * shape. `input_text` is rewritten to `text` (the part the existing prompt
 * formatter understands); image-bearing parts are preserved verbatim so the
 * vision extractor can pick them up.
 */
function normalizeContent(content: any): string | OpenAIContentPart[] | undefined {
  if (content == null) return undefined;
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return undefined;

  const parts: OpenAIContentPart[] = [];
  for (const p of content) {
    if (!p || typeof p !== 'object') continue;
    switch (p.type) {
      case 'input_text':
      case 'text':
      case 'output_text':
        parts.push({ type: 'text', text: String(p.text ?? '') } as OpenAIContentPart);
        break;
      case 'refusal':
        parts.push({ type: 'text', text: String(p.refusal ?? '') } as OpenAIContentPart);
        break;
      case 'input_image':
      case 'image_url':
      case 'image':
      case 'input_file':
        parts.push(p as OpenAIContentPart);
        break;
      default:
        break;
    }
  }
  return parts.length ? parts : undefined;
}

/**
 * Convert the Responses `input` field into the message list the chat pipeline
 * expects. `input` may be a bare string or an array of items.
 */
function normalizeInput(input: ResponsesRequest['input']): OpenAIMessage[] {
  if (input == null) return [];
  if (typeof input === 'string') {
    return [{ role: 'user', content: input } as OpenAIMessage];
  }
  if (!Array.isArray(input)) return [];

  const messages: OpenAIMessage[] = [];
  const bareParts: OpenAIContentPart[] = [];

  for (const item of input as any[]) {
    if (!item || typeof item !== 'object') continue;
    const type = item.type;

    if (type === 'function_call') {
      const args =
        typeof item.arguments === 'string'
          ? item.arguments
          : JSON.stringify(item.arguments ?? {});
      messages.push({
        role: 'assistant',
        content: '',
        tool_calls: [
          {
            id: item.call_id || item.id || `call_${crypto.randomUUID()}`,
            type: 'function',
            function: { name: item.name || '', arguments: args },
          },
        ],
      } as any);
      continue;
    }

    if (type === 'function_call_output') {
      messages.push({
        role: 'tool',
        tool_call_id: item.call_id || item.id || '',
        content:
          typeof item.output === 'string'
            ? item.output
            : JSON.stringify(item.output ?? ''),
      } as any);
      continue;
    }

    // Reasoning items are replayed state, not new input.
    if (type === 'reasoning') continue;

    if (type === 'message' || item.role) {
      const content = normalizeContent(item.content);
      if (content !== undefined) {
        messages.push({ role: (item.role || 'user') as any, content } as OpenAIMessage);
      }
      continue;
    }

    // A bare content part (input_text / input_image / input_file) at the top
    // level belongs to the implicit user turn.
    if (partIsImage(item) || type === 'input_text' || type === 'text') {
      const normalized = normalizeContent([item]);
      if (Array.isArray(normalized)) bareParts.push(...normalized);
      else if (typeof normalized === 'string') {
        bareParts.push({ type: 'text', text: normalized } as OpenAIContentPart);
      }
    }
  }

  if (bareParts.length) {
    messages.push({ role: 'user', content: collapseParts(bareParts) } as OpenAIMessage);
  }
  return messages;
}

/**
 * Responses tools are flat (`{type, name, description, parameters}`); the chat
 * pipeline is proven against the nested chat-completions shape, so flatten
 * nothing and nest instead.
 */
function normalizeTools(tools: any[] | undefined): OpenAITool[] {
  if (!Array.isArray(tools)) return [];
  const out: OpenAITool[] = [];
  for (const t of tools) {
    if (!t || typeof t !== 'object') continue;
    if (t.function?.name) {
      out.push(t as OpenAITool);
      continue;
    }
    if (!t.name) continue;
    const parameters =
      t.parameters || t.format?.schema || { type: 'object', properties: {} };
    out.push({
      type: 'function',
      function: { name: t.name, description: t.description, parameters },
    } as any);
  }
  return out;
}

/* ------------------------------------------------------------------ *
 * Chat-pipeline adapter plumbing
 * ------------------------------------------------------------------ */

/**
 * A minimal `ServerResponse` stand-in that captures everything the chat
 * controller writes instead of sending it to the client.
 */
function createCaptureResponse(onChunk: (text: string) => void, onEnd?: () => void): any {
  const capture: any = {
    headersSent: false,
    writableEnded: false,
    finished: false,
    statusCode: 200,
    _headers: {} as Record<string, any>,
    writeHead(status: number, headers?: any) {
      capture.statusCode = status;
      capture.headersSent = true;
      if (headers && typeof headers === 'object') Object.assign(capture._headers, headers);
      return capture;
    },
    setHeader(name: string, value: any) {
      capture._headers[String(name).toLowerCase()] = value;
    },
    getHeader(name: string) {
      return capture._headers[String(name).toLowerCase()];
    },
    removeHeader(name: string) {
      delete capture._headers[String(name).toLowerCase()];
    },
    flushHeaders() {},
    write(chunk: any) {
      if (chunk != null) {
        onChunk(typeof chunk === 'string' ? chunk : Buffer.from(chunk).toString('utf8'));
      }
      return true;
    },
    end(chunk?: any) {
      if (chunk != null) capture.write(chunk);
      capture.headersSent = true;
      capture.writableEnded = true;
      capture.finished = true;
      if (onEnd) {
        try {
          onEnd();
        } catch {
          /* the end signal must never mask the captured response */
        }
      }
      return capture;
    },
    on() { return capture; },
    once() { return capture; },
    off() { return capture; },
    removeListener() { return capture; },
    emit() { return false; },
  };
  return capture;
}

/**
 * Wrap the synthesized chat body in a request-like stream that preserves the
 * original headers (the chat controller reads the bearer token from them).
 */
function makeFakeRequest(realReq: IncomingMessage, body: string): IncomingMessage {
  const stream = new PassThrough() as any;
  stream.headers = realReq.headers;
  stream.method = realReq.method;
  stream.url = realReq.url;
  stream.httpVersion = realReq.httpVersion;
  stream.socket = (realReq as any).socket;
  stream.connection = (realReq as any).connection;
  process.nextTick(() => {
    stream.write(Buffer.from(body, 'utf8'));
    stream.end();
  });
  return stream as IncomingMessage;
}

/* ------------------------------------------------------------------ *
 * SSE translation: chat.completion.chunk -> response.* events
 * ------------------------------------------------------------------ */

type Emit = (event: Record<string, any>) => void;

interface ToolSlot {
  outputIndex: number;
  id: string;
  callId: string;
  name: string;
  args: string;
}

/**
 * Translates the proxy's `chat.completion.chunk` SSE stream into the OpenAI
 * Responses `response.*` event sequence that pi-ai's parser expects:
 * `output_item.added` (slot creation, typed by `item.type`), per-kind deltas
 * keyed by `output_index`, then `output_item.done` and `response.completed`.
 */
class ResponsesStreamTranslator {
  private buffer = '';
  private readonly responseId: string;
  private readonly createdAt: number;
  private created = false;
  private nextIndex = 0;
  private reasoning: { index: number; id: string; text: string } | null = null;
  private message: { index: number; id: string; text: string } | null = null;
  private readonly toolCalls = new Map<number, ToolSlot>();
  private usage: any = null;
  private finalized = false;

  constructor(private readonly model: string, private readonly emit: Emit) {
    this.responseId = `resp_${crypto.randomUUID().replace(/-/g, '')}`;
    this.createdAt = Math.floor(Date.now() / 1000);
  }

  /** Feed raw SSE text; frames are split on the blank-line delimiter. */
  push(chunk: string): void {
    this.buffer += chunk;
    let idx: number;
    while ((idx = this.buffer.indexOf('\n\n')) !== -1) {
      const frame = this.buffer.slice(0, idx);
      this.buffer = this.buffer.slice(idx + 2);
      this.handleFrame(frame);
    }
  }

  private handleFrame(frame: string): void {
    for (const line of frame.split('\n')) {
      if (!line.startsWith('data:')) continue;
      const payload = line.slice(5).trim();
      if (!payload || payload === '[DONE]') continue;
      let obj: any;
      try {
        obj = JSON.parse(payload);
      } catch {
        continue;
      }
      this.handleChunk(obj);
    }
  }

  private handleChunk(obj: any): void {
    if (obj && obj.usage) this.usage = obj.usage;
    const choice = obj?.choices?.[0];
    if (!choice) return;

    const delta = choice.delta || {};
    const reasoningDelta =
      typeof delta.reasoning_content === 'string'
        ? delta.reasoning_content
        : typeof delta.reasoning === 'string'
          ? delta.reasoning
          : '';
    if (reasoningDelta) this.appendReasoning(reasoningDelta);

    if (typeof delta.content === 'string' && delta.content) {
      this.appendText(delta.content);
    }

    if (Array.isArray(delta.tool_calls)) {
      for (const tc of delta.tool_calls) this.appendToolCall(tc);
    }
  }

  private ensureCreated(): void {
    if (this.created) return;
    this.created = true;
    this.emit({
      type: 'response.created',
      response: {
        id: this.responseId,
        object: 'response',
        created_at: this.createdAt,
        status: 'in_progress',
        model: this.model,
        output: [],
      },
    });
  }

  private appendReasoning(text: string): void {
    this.ensureCreated();
    if (!this.reasoning) {
      const index = this.nextIndex++;
      const id = `rs_${crypto.randomUUID().replace(/-/g, '')}`;
      this.reasoning = { index, id, text: '' };
      this.emit({
        type: 'response.output_item.added',
        output_index: index,
        item: { type: 'reasoning', id, summary: [] },
      });
    }
    this.reasoning.text += text;
    this.emit({
      type: 'response.reasoning_summary_text.delta',
      output_index: this.reasoning.index,
      item_id: this.reasoning.id,
      summary_index: 0,
      delta: text,
    });
  }

  private appendText(text: string): void {
    this.ensureCreated();
    if (!this.message) {
      const index = this.nextIndex++;
      const id = `msg_${crypto.randomUUID().replace(/-/g, '')}`;
      this.message = { index, id, text: '' };
      this.emit({
        type: 'response.output_item.added',
        output_index: index,
        item: { type: 'message', id, status: 'in_progress', role: 'assistant', content: [] },
      });
    }
    this.message.text += text;
    this.emit({
      type: 'response.output_text.delta',
      output_index: this.message.index,
      item_id: this.message.id,
      content_index: 0,
      delta: text,
    });
  }

  private appendToolCall(tc: any): void {
    this.ensureCreated();
    const key = typeof tc?.index === 'number' ? tc.index : 0;
    let slot = this.toolCalls.get(key);
    if (!slot) {
      const outputIndex = this.nextIndex++;
      const callId = tc?.id || `call_${crypto.randomUUID().replace(/-/g, '')}`;
      slot = {
        outputIndex,
        id: `fc_${crypto.randomUUID().replace(/-/g, '')}`,
        callId,
        name: tc?.function?.name || '',
        args: '',
      };
      this.toolCalls.set(key, slot);
      this.emit({
        type: 'response.output_item.added',
        output_index: outputIndex,
        item: {
          type: 'function_call',
          id: slot.id,
          call_id: slot.callId,
          name: slot.name,
          arguments: '',
        },
      });
    }
    if (!slot.name && tc?.function?.name) slot.name = tc.function.name;

    const argDelta = tc?.function?.arguments;
    if (typeof argDelta === 'string' && argDelta.length) {
      slot.args += argDelta;
      this.emit({
        type: 'response.function_call_arguments.delta',
        output_index: slot.outputIndex,
        item_id: slot.id,
        delta: argDelta,
      });
    }
  }

  /** Close every open item and emit the terminal `response.completed`. */
  finish(): void {
    if (this.finalized) return;
    this.finalized = true;
    this.ensureCreated();

    const output: any[] = [];

    if (this.reasoning) {
      const item = {
        type: 'reasoning',
        id: this.reasoning.id,
        summary: this.reasoning.text
          ? [{ type: 'summary_text', text: this.reasoning.text }]
          : [],
      };
      this.emit({ type: 'response.output_item.done', output_index: this.reasoning.index, item });
      output.push(item);
    }

    if (this.message) {
      const item = {
        type: 'message',
        id: this.message.id,
        status: 'completed',
        role: 'assistant',
        content: this.message.text
          ? [{ type: 'output_text', text: this.message.text, annotations: [] }]
          : [],
      };
      this.emit({ type: 'response.output_item.done', output_index: this.message.index, item });
      output.push(item);
    }

    for (const slot of [...this.toolCalls.values()].sort((a, b) => a.outputIndex - b.outputIndex)) {
      const item = {
        type: 'function_call',
        id: slot.id,
        call_id: slot.callId,
        name: slot.name,
        arguments: slot.args,
      };
      this.emit({
        type: 'response.function_call_arguments.done',
        output_index: slot.outputIndex,
        item_id: slot.id,
        arguments: slot.args,
      });
      this.emit({ type: 'response.output_item.done', output_index: slot.outputIndex, item });
      output.push(item);
    }

    this.emit({
      type: 'response.completed',
      response: {
        id: this.responseId,
        object: 'response',
        created_at: this.createdAt,
        status: 'completed',
        model: this.model,
        output,
        usage: this.buildUsage(),
      },
    });
  }

  /** Emit a terminal error event; pi-ai surfaces this as a thrown error. */
  emitError(message: string): void {
    if (this.finalized) return;
    this.finalized = true;
    this.ensureCreated();
    this.emit({ type: 'error', code: 'proxy_error', message });
  }

  private buildUsage(): any {
    if (!this.usage) return undefined;
    return {
      input_tokens: this.usage.prompt_tokens ?? 0,
      output_tokens: this.usage.completion_tokens ?? 0,
      total_tokens: this.usage.total_tokens ?? 0,
      output_tokens_details: {
        reasoning_tokens: this.usage.completion_tokens_details?.reasoning_tokens ?? 0,
      },
    };
  }
}

/* ------------------------------------------------------------------ *
 * Entry point
 * ------------------------------------------------------------------ */

export async function handleResponses(
  req: IncomingMessage,
  res: ServerResponse,
  userToken: string,
): Promise<void> {
  let bodyText: string;
  try {
    bodyText = await readBody(req);
  } catch (err: any) {
    sendJsonError(res, 400, `Failed to read request body: ${err?.message || err}`, 'invalid_request_error');
    return;
  }

  let parsed: ResponsesRequest;
  try {
    parsed = JSON.parse(bodyText || '{}');
  } catch {
    sendJsonError(res, 400, 'Invalid JSON', 'invalid_request_error');
    return;
  }

  const model = typeof parsed.model === 'string' && parsed.model ? parsed.model : 'deepseek-chat';

  const messages = normalizeInput(parsed.input);
  if (typeof parsed.instructions === 'string' && parsed.instructions) {
    messages.unshift({ role: 'system', content: parsed.instructions } as OpenAIMessage);
  }
  if (messages.length === 0) {
    sendJsonError(res, 400, '`input` must contain at least one message', 'invalid_request_error');
    return;
  }

  const tools = normalizeTools(parsed.tools as any[] | undefined);

  // The chat pipeline is always driven in streaming mode: the proxy may force
  // it anyway, and a single code path keeps both Responses modes identical.
  const chatRequest: Record<string, any> = {
    model,
    messages,
    stream: true,
    thinking_enabled: (parsed as any).thinking_enabled,
    chat_session_id: (parsed as any).chat_session_id,
  };
  if (tools.length) {
    chatRequest.tools = tools;
    if (parsed.tool_choice !== undefined) chatRequest.tool_choice = parsed.tool_choice;
  }
  if (parsed.max_output_tokens !== undefined) chatRequest.max_tokens = parsed.max_output_tokens;
  if (parsed.temperature !== undefined) chatRequest.temperature = parsed.temperature;
  if (parsed.top_p !== undefined) chatRequest.top_p = parsed.top_p;

  const chatBody = JSON.stringify(chatRequest);

  // Streaming: translate the chat SSE into Responses SSE as it arrives.
  if ((parsed as any).stream === true) {
    res.writeHead(200, {
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
      'X-Accel-Buffering': 'no',
    });

    let signalStreamEnd: () => void = () => {};
    const streamEnded = new Promise<void>((resolve) => {
      signalStreamEnd = resolve;
    });

    const translator = new ResponsesStreamTranslator(model, (event) => {
      if (!res.writableEnded) {
        res.write(`event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`);
      }
    });
    const capture = createCaptureResponse(
      (text) => translator.push(text),
      () => signalStreamEnd(),
    );

    try {
      await handleChatCompletions(
        makeFakeRequest(req, chatBody),
        capture as unknown as ServerResponse,
        userToken,
      );
      // `handleChatCompletions` resolves as soon as it hands the upstream
      // stream to its pump, not when that stream drains. The captured
      // `end()` is the real terminal signal; without waiting for it the
      // translator would finalize on an empty buffer and drop every chunk.
      await streamEnded;
      translator.finish();
    } catch (err: any) {
      translator.emitError(err?.message || 'Proxy failure');
    }

    if (!res.writableEnded) res.end();
    return;
  }

  // Non-streaming: replay the same pipeline, then return only the terminal
  // response object, so both modes share one implementation.
  const events: Record<string, any>[] = [];
  let signalStreamEnd: () => void = () => {};
  const streamEnded = new Promise<void>((resolve) => {
    signalStreamEnd = resolve;
  });
  const translator = new ResponsesStreamTranslator(model, (event) => events.push(event));
  const capture = createCaptureResponse(
    (text) => translator.push(text),
    () => signalStreamEnd(),
  );

  try {
    await handleChatCompletions(
      makeFakeRequest(req, chatBody),
      capture as unknown as ServerResponse,
      userToken,
    );
    // See the streaming branch: the capture's `end()` is the real terminal
    // signal, and the chat controller returns well before it fires.
    await streamEnded;
    translator.finish();
  } catch (err: any) {
    sendJsonError(res, 502, err?.message || 'Proxy failure', 'bad_gateway');
    return;
  }

  const failure = events.find((e) => e.type === 'error');
  if (failure) {
    sendJsonError(res, 502, failure.message || 'Upstream failure', 'bad_gateway');
    return;
  }

  const completed = events.find((e) => e.type === 'response.completed');
  if (!completed) {
    sendJsonError(res, 502, 'Upstream produced no response', 'bad_gateway');
    return;
  }

  sendJsonResponse(res, 200, completed.response);
}
