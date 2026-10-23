export interface OpenAIMessage {
  role: 'system' | 'user' | 'assistant' | 'tool' | 'function' | string;
  content: string | OpenAIContentPart[] | any;
  name?: string;
  tool_call_id?: string;
  tool_calls?: OpenAIToolCall[];
  function_call?: {
    name: string;
    arguments: string;
  };
  reasoning_content?: string;
  thought?: string;
}

/**
 * Anthropic-style image source. Anthropic Messages sends
 * `{type:'base64', media_type, data}` or `{type:'url', url}`; some clients
 * reuse the same shape inside OpenAI-ish payloads.
 */
export interface ImageSourceObject {
  type?: 'base64' | 'url' | string;
  media_type?: string;
  /** base64 payload WITHOUT the `data:` prefix, when type === 'base64'. */
  data?: string;
  /** Remote URL, when type === 'url'. */
  url?: string;
}

/**
 * OpenAI Responses API image part. The image travels as bytes inside the
 * request: either a `data:` URL in `image_url`, a remote URL, or a server-side
 * `file_id` reference.
 */
export interface InputImagePart {
  type: 'input_image';
  image_url?: string;
  file_id?: string;
  detail?: 'auto' | 'low' | 'high' | string;
}

/**
 * OpenAI Responses API file part. `file_data` is base64 (optionally wrapped as
 * a `data:<mime>;base64,<payload>` URL); `file_id` references a server file.
 */
export interface InputFilePart {
  type: 'input_file';
  file_data?: string;
  file_id?: string;
  filename?: string;
}

export type OpenAIContentPart =
  | { type: 'text' | 'input_text'; text: string }
  | { type: 'image_url'; image_url: string | { url: string; detail?: string } }
  | { type: 'image'; image?: string; source?: string | ImageSourceObject }
  | InputImagePart
  | InputFilePart
  | { type: 'tool_use' | 'tool_call'; id?: string; name?: string; input?: unknown }
  | { type: 'tool_result'; content: string | object };

export interface OpenAITool {
  type?: string;
  function?: {
    name: string;
    description?: string;
    parameters?: Record<string, unknown>;
  };
  name?: string;
  description?: string;
  parameters?: Record<string, unknown>;
  input_schema?: Record<string, unknown>;
}

export interface OpenAIToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    arguments: string;
  };
}

export interface ChatCompletionRequest {
  model?: string;
  messages: OpenAIMessage[];
  stream?: boolean;
  tools?: OpenAITool[];
  functions?: OpenAITool[];
  chat_session_id?: string;
  thinking_enabled?: boolean;
  reasoning?: boolean;
  model_type?: 'default' | string;
  /** The client sends its own system prompt and workflow (see PromptFormatterService.format). */
  client_prompt?: boolean;
}

// ---------------------------------------------------------------------------
// OpenAI Responses API (`POST /v1/responses`)
// ---------------------------------------------------------------------------

/** One item of a Responses API `input` array. */
export interface ResponsesInputItem {
  type?: string;
  role?: string;
  content?: string | OpenAIContentPart[];
  /** `function_call_output` correlation id. */
  call_id?: string;
  /** `function_call_output` payload (text or multimodal parts). */
  output?: string | OpenAIContentPart[];
  [key: string]: unknown;
}

export interface ResponsesRequest {
  model?: string;
  /** A bare string, or the itemized form carrying text/image/file parts. */
  input?: string | ResponsesInputItem[];
  instructions?: string;
  stream?: boolean;
  tools?: any[];
  tool_choice?: unknown;
  max_output_tokens?: number;
  temperature?: number;
  top_p?: number;
  reasoning?: { effort?: string; summary?: string };
  chat_session_id?: string;
  thinking_enabled?: boolean;
  [key: string]: unknown;
}

export interface TokenUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  totalTokens?: number;
  inputTokens?: number;
  outputTokens?: number;
  input?: number;
  output?: number;
}

export interface ChatCompletionResponseChoice {
  index: number;
  message: {
    role: 'assistant';
    content: string | null;
    reasoning_content?: string;
    reasoning?: string;
    tool_calls?: OpenAIToolCall[];
  };
  finish_reason: 'stop' | 'tool_calls' | 'length' | null;
}

export interface ChatCompletionResponse {
  id: string;
  object: 'chat.completion';
  created: number;
  model: string;
  choices: ChatCompletionResponseChoice[];
  usage?: TokenUsage;
}
