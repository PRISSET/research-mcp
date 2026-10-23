import path from 'path';

const targetHost = process.env.TARGET_HOST || 'chat.deepseek.com';
const clientHeaders = {
  Origin: `https://${targetHost}`,
  Referer: `https://${targetHost}/`,
  'User-Agent':
    process.env.USER_AGENT ||
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
  'x-app-version': process.env.APP_VERSION || '2.3.0',
  'x-client-platform': process.env.CLIENT_PLATFORM || 'web',
  'x-client-version': process.env.CLIENT_VERSION || '2.3.0',
  'x-client-locale': process.env.CLIENT_LOCALE || 'en_US',
};

export const config = {
  port: parseInt(process.env.PROXY_PORT || '3085', 10),
  timeoutMs: parseInt(process.env.REQUEST_TIMEOUT_MS || '120000', 10),
  targetHost,
  clientHeaders,
  wasmPath: path.join(__dirname, '..', '..', 'sha3_wasm.wasm'),
  image: {
    convertCmd: process.env.IMAGE_CONVERT_CMD || 'cwebp -q 85',
    tempDir: process.env.IMAGE_TEMP_DIR || '/tmp',
    maxSize: parseInt(process.env.IMAGE_MAX_SIZE || '20', 10) * 1024 * 1024,
    uploadRetries: parseInt(process.env.IMAGE_UPLOAD_RETRIES || '5', 10),
    uploadRetryDelay: parseInt(process.env.IMAGE_UPLOAD_RETRY_DELAY || '800', 10),
  },
  models: {
    default: {
      id: 'deepseek-chat',
      contextWindow: parseInt(process.env.MODEL_DEFAULT_CONTEXT_WINDOW || '64000', 10),
      maxTokens: parseInt(process.env.MODEL_DEFAULT_MAX_TOKENS || '8192', 10),
    },
  },
  pow: {
    jsFallbackLimit: parseInt(process.env.POW_JS_LIMIT || '10000000', 10),
  },
  forceStream: process.env.FORCE_STREAM === 'true',
  // Authorization-token rotation. The harness (or any OpenAI-compatible
  // client) may supply a pipe-delimited list of DeepSeek web tokens as the
  // bearer value: `token1|token2|token3`. Each incoming request is served by
  // one token, selected round-robin, so a single token never absorbs the whole
  // request rate. A token that answers with a throttle status (429/503/529) is
  // put in cooldown (exponential, capped) and skipped until it expires.
  authToken: {
    rotationEnabled: process.env.TOKEN_ROTATION_ENABLED !== 'false',
    cooldownMs: parseInt(process.env.TOKEN_COOLDOWN_MS || '60000', 10),
    maxCooldownMs: parseInt(process.env.TOKEN_MAX_COOLDOWN_MS || '900000', 10),
    maxRetries: parseInt(process.env.TOKEN_MAX_RETRIES || '3', 10),
  },
  token: {
    // Chars-per-token ratios calibrated to DeepSeek's published guidance
    // (~0.3 token per English char, ~0.6 token per CJK char) plus observed
    // behaviour for Cyrillic and code/JSON. Every ratio is env-overridable so
    // an operator can re-calibrate without a code change.
    asciiCharsPerToken: parseFloat(process.env.TOKEN_ASCII_CPT || '3.6'),
    cyrillicCharsPerToken: parseFloat(process.env.TOKEN_CYRILLIC_CPT || '2.0'),
    cjkCharsPerToken: parseFloat(process.env.TOKEN_CJK_CPT || '1.67'),
    punctuationCharsPerToken: parseFloat(process.env.TOKEN_PUNCT_CPT || '1.4'),
    whitespaceCharsPerToken: parseFloat(process.env.TOKEN_WS_CPT || '3.0'),
    emojiTokensPerChar: parseFloat(process.env.TOKEN_EMOJI_TPC || '2.5'),
    otherCharsPerToken: parseFloat(process.env.TOKEN_OTHER_CPT || '2.5'),
    // A single space is folded into the following word token by BPE, so it
    // costs almost nothing; a lone newline or tab is close to a full token.
    // These two dominate the estimate for prose, hence env-overridable.
    singleSpaceTokens: parseFloat(process.env.TOKEN_SINGLE_SPACE || '0.05'),
    newlineTokens: parseFloat(process.env.TOKEN_NEWLINE || '0.5'),
    messageOverhead: parseInt(process.env.TOKEN_MESSAGE_OVERHEAD || '4', 10),
    promptOverhead: parseInt(process.env.TOKEN_PROMPT_OVERHEAD || '3', 10),
  },
  usage: {
    // DeepSeek's web SSE only exposes `accumulated_token_usage`. Default
    // assumption: it counts completion/output tokens only. Set
    // ACCUMULATED_IS_TOTAL=true if it is actually prompt+completion, which
    // lets prompt_tokens be derived from a real measured value.
    accumulatedIsTotal: process.env.ACCUMULATED_IS_TOTAL === 'true',
    // Append a final usage chunk to every stream (many harnesses require it)
    // even when the client did not request it via stream_options.include_usage.
    alwaysIncludeStreamUsage: process.env.STREAM_USAGE_ALWAYS === 'true',
  },
};
