/**
 * Offline, deterministic token estimator for the DeepSeek Chat Web Bridge.
 *
 * WHY THIS EXISTS
 * ---------------
 * DeepSeek's web SSE endpoint never returns an input/prompt token count: it
 * only exposes `accumulated_token_usage`, which the parser maps to the
 * completion side. The controller therefore had to guess the prompt size with
 * `Math.ceil(prompt.length / 4)`. That heuristic is wrong in both directions:
 *
 *   - it UNDER-counts Cyrillic (~1.5-2 chars/token), code and JSON, where
 *     punctuation and identifiers rarely merge into a single BPE token;
 *   - it OVER-counts long ASCII prose, where common words become one token.
 *
 * A harness that trusts those numbers either compacts too late (context
 * overflow / silent truncation) or too early (wasted budget). This module
 * replaces the flat `len / 4` guess with a Unicode-class-aware estimator
 * calibrated to DeepSeek's published tokenization guidance:
 *
 *   ~0.3 token per English character  -> ~3.6 chars/token
 *   ~0.6 token per Chinese character  -> ~1.67 chars/token
 *
 * It is intentionally dependency-free and synchronous so it can run on every
 * request without touching the network or the event loop. The public surface
 * is small and stable so a real BPE tokenizer (e.g. DeepSeek's
 * `tokenizer.json`) can be swapped in behind it later without touching callers.
 */

import { config } from '../../config';

/** Tunable ratios. `charsPerToken` means "how many characters make one token". */
export interface TokenEstimatorConfig {
  asciiCharsPerToken: number;
  cyrillicCharsPerToken: number;
  cjkCharsPerToken: number;
  punctuationCharsPerToken: number;
  whitespaceCharsPerToken: number;
  emojiTokensPerChar: number;
  otherCharsPerToken: number;
  /** Cost of a lone space (BPE folds it into the next token). */
  singleSpaceTokens: number;
  /** Cost of a lone newline or tab. */
  newlineTokens: number;
  /** Fixed per-message overhead for role/formatting markers. */
  messageOverhead: number;
  /** Fixed overhead for the chat scaffold / reply priming. */
  promptOverhead: number;
}

/**
 * Defaults match `config.token` so the estimator also works standalone
 * (unit tests, scripts) without booting the full config.
 */
export const DEFAULT_TOKEN_ESTIMATOR_CONFIG: TokenEstimatorConfig = {
  asciiCharsPerToken: 3.6,
  cyrillicCharsPerToken: 2.0,
  cjkCharsPerToken: 1.67,
  punctuationCharsPerToken: 1.4,
  whitespaceCharsPerToken: 3.0,
  emojiTokensPerChar: 2.5,
  otherCharsPerToken: 2.5,
  singleSpaceTokens: 0.05,
  newlineTokens: 0.5,
  messageOverhead: 4,
  promptOverhead: 3,
};

type CharClass = 'ascii' | 'cyrillic' | 'cjk' | 'punctuation' | 'whitespace' | 'emoji' | 'other';

/** Unicode code-point classifier. Ranges are intentionally generous. */
export function classifyCodePoint(cp: number): CharClass {
  // Whitespace (incl. NBSP and the ideographic space).
  if (cp === 0x20 || cp === 0x09 || cp === 0x0a || cp === 0x0d || cp === 0x0b || cp === 0x0c) return 'whitespace';
  if (cp === 0xa0 || cp === 0x1680 || (cp >= 0x2000 && cp <= 0x200a) || cp === 0x2028 || cp === 0x2029 || cp === 0x202f || cp === 0x205f || cp === 0x3000) return 'whitespace';

  // CJK: kana, CJK ideographs + extensions, Hangul, Bopomofo, fullwidth forms.
  if (cp >= 0x3001 && cp <= 0x303f) return 'cjk';
  if (cp >= 0x3040 && cp <= 0x30ff) return 'cjk';
  if (cp >= 0x31f0 && cp <= 0x31ff) return 'cjk';
  if (cp >= 0x3400 && cp <= 0x4dbf) return 'cjk';
  if (cp >= 0x4e00 && cp <= 0x9fff) return 'cjk';
  if (cp >= 0xf900 && cp <= 0xfaff) return 'cjk';
  if (cp >= 0xac00 && cp <= 0xd7af) return 'cjk';
  if (cp >= 0x1100 && cp <= 0x11ff) return 'cjk';
  if (cp >= 0x20000 && cp <= 0x2fa1f) return 'cjk';
  if (cp >= 0xff00 && cp <= 0xffef) return 'cjk';

  // Cyrillic (incl. supplement, extended-A and extended-B).
  if (cp >= 0x0400 && cp <= 0x04ff) return 'cyrillic';
  if (cp >= 0x0500 && cp <= 0x052f) return 'cyrillic';
  if (cp >= 0x2de0 && cp <= 0x2dff) return 'cyrillic';
  if (cp >= 0xa640 && cp <= 0xa69f) return 'cyrillic';

  // Emoji / pictographs / dingbats / symbols.
  if (cp >= 0x1f000 && cp <= 0x1faff) return 'emoji';
  if (cp >= 0x2600 && cp <= 0x27bf) return 'emoji';
  if (cp === 0xfe0f || cp === 0x200d) return 'emoji';
  if (cp >= 0x1f1e6 && cp <= 0x1f1ff) return 'emoji';
  if (cp > 0xffff) return 'emoji';

  // ASCII / Latin letters and digits: the class BPE merges most aggressively.
  if (cp >= 0x30 && cp <= 0x39) return 'ascii';
  if (cp >= 0x41 && cp <= 0x5a) return 'ascii';
  if (cp >= 0x61 && cp <= 0x7a) return 'ascii';
  if (cp === 0x5f) return 'ascii';
  if (cp >= 0xc0 && cp <= 0x24f) return 'ascii';

  return 'punctuation';
}

/**
 * Estimate tokens for an arbitrary string. Pure and synchronous.
 *
 * Whitespace runs are charged a minimum of one token, since a leading space is
 * normally folded into the following token rather than billed separately.
 */
export function countTokens(text: string, cfg: TokenEstimatorConfig = DEFAULT_TOKEN_ESTIMATOR_CONFIG): number {
  if (!text) return 0;

  let ascii = 0;
  let cyrillic = 0;
  let cjk = 0;
  let punctuation = 0;
  let whitespace = 0;
  let emoji = 0;
  let other = 0;

  for (const ch of text) {
    const cp = ch.codePointAt(0);
    if (cp === undefined) continue;
    switch (classifyCodePoint(cp)) {
      case 'ascii': ascii++; break;
      case 'cyrillic': cyrillic++; break;
      case 'cjk': cjk++; break;
      case 'punctuation': punctuation++; break;
      case 'whitespace': whitespace++; break;
      case 'emoji': emoji++; break;
      default: other++;
    }
  }

  const safe = (n: number) => (n > 0 ? n : 1);
  let tokens = 0;
  tokens += ascii / safe(cfg.asciiCharsPerToken);
  tokens += cyrillic / safe(cfg.cyrillicCharsPerToken);
  tokens += cjk / safe(cfg.cjkCharsPerToken);
  tokens += punctuation / safe(cfg.punctuationCharsPerToken);
  tokens += other / safe(cfg.otherCharsPerToken);
  tokens += emoji * cfg.emojiTokensPerChar;
  // Whitespace: BPE folds a single leading space into the following word
  // token, so an isolated space is nearly free. Charging a full token per run
  // (the obvious `runs.length` heuristic) roughly doubles the estimate for
  // English prose, which then triggers compaction far too early. Only
  // multi-character runs -- indentation, alignment, blank lines -- are
  // charged, plus a flat cost for lone newlines/tabs.
  if (whitespace > 0) {
    const runs = text.match(/\s+/g) || [];
    for (const run of runs) {
      if (run.length === 1) {
        tokens += run === ' ' ? cfg.singleSpaceTokens : cfg.newlineTokens;
      } else {
        tokens += 1 + (run.length - 1) / safe(cfg.whitespaceCharsPerToken);
      }
    }
  }

  return Math.ceil(tokens);
}

/**
 * Count the formatted prompt that is actually sent to DeepSeek.
 *
 * This is the ground truth for context-window pressure: it includes the
 * system scaffold and the serialized tool schemas that the proxy injects.
 */
export function countPromptTokens(formattedPrompt: string, cfg: TokenEstimatorConfig = DEFAULT_TOKEN_ESTIMATOR_CONFIG): number {
  if (!formattedPrompt) return cfg.promptOverhead;
  return countTokens(formattedPrompt, cfg) + cfg.promptOverhead;
}

/** Pull a plain-text representation out of a message's `content`. */
function extractContentText(content: unknown): string {
  if (content == null) return '';
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const parts: string[] = [];
    for (const part of content) {
      if (typeof part === 'string') {
        parts.push(part);
      } else if (part && typeof part === 'object') {
        const p = part as Record<string, any>;
        if (typeof p.text === 'string') parts.push(p.text);
        else if (typeof p.content === 'string') parts.push(p.content);
        else if (p.image_url || p.type === 'image' || p.type === 'image_url') parts.push('[image]');
        else if (p.type === 'tool_result' || p.type === 'tool_use') parts.push(JSON.stringify(p.input ?? p.content ?? p));
      }
    }
    return parts.join('\n');
  }
  if (typeof content === 'object') {
    try {
      return JSON.stringify(content);
    } catch {
      return String(content);
    }
  }
  return String(content);
}

/**
 * Estimate the tokens of the harness-supplied context (raw messages + tools),
 * i.e. what the client *thinks* it sent, before the proxy adds its scaffold.
 *
 * Useful for diagnostics and for exposing an input count the harness can
 * reconcile against its own accounting.
 */
export function countMessageTokens(
  messages: Array<Record<string, any>> = [],
  tools: Array<Record<string, any>> = [],
  cfg: TokenEstimatorConfig = DEFAULT_TOKEN_ESTIMATOR_CONFIG
): number {
  let tokens = cfg.promptOverhead;

  for (const msg of messages) {
    if (!msg) continue;
    tokens += cfg.messageOverhead;
    tokens += countTokens(extractContentText(msg.content), cfg);
    if (typeof msg.name === 'string' && msg.name) tokens += countTokens(msg.name, cfg);
    if (typeof msg.tool_call_id === 'string' && msg.tool_call_id) tokens += countTokens(msg.tool_call_id, cfg);
    if (msg.tool_calls) {
      try {
        tokens += countTokens(JSON.stringify(msg.tool_calls), cfg);
      } catch { /* ignore unserializable */ }
    }
    if (msg.function_call) {
      try {
        tokens += countTokens(JSON.stringify(msg.function_call), cfg);
      } catch { /* ignore unserializable */ }
    }
  }

  if (Array.isArray(tools) && tools.length > 0) {
    try {
      tokens += countTokens(JSON.stringify(tools), cfg);
    } catch { /* ignore unserializable */ }
  }

  return Math.ceil(tokens);
}

/** Resolve the estimator config from the running application config. */
export function getEstimatorConfig(): TokenEstimatorConfig {
  const t: any = (config as any).token;
  if (!t) return DEFAULT_TOKEN_ESTIMATOR_CONFIG;
  return {
    asciiCharsPerToken: numOr(t.asciiCharsPerToken, DEFAULT_TOKEN_ESTIMATOR_CONFIG.asciiCharsPerToken),
    cyrillicCharsPerToken: numOr(t.cyrillicCharsPerToken, DEFAULT_TOKEN_ESTIMATOR_CONFIG.cyrillicCharsPerToken),
    cjkCharsPerToken: numOr(t.cjkCharsPerToken, DEFAULT_TOKEN_ESTIMATOR_CONFIG.cjkCharsPerToken),
    punctuationCharsPerToken: numOr(t.punctuationCharsPerToken, DEFAULT_TOKEN_ESTIMATOR_CONFIG.punctuationCharsPerToken),
    whitespaceCharsPerToken: numOr(t.whitespaceCharsPerToken, DEFAULT_TOKEN_ESTIMATOR_CONFIG.whitespaceCharsPerToken),
    emojiTokensPerChar: numOr(t.emojiTokensPerChar, DEFAULT_TOKEN_ESTIMATOR_CONFIG.emojiTokensPerChar),
    otherCharsPerToken: numOr(t.otherCharsPerToken, DEFAULT_TOKEN_ESTIMATOR_CONFIG.otherCharsPerToken),
    singleSpaceTokens: numOr(t.singleSpaceTokens, DEFAULT_TOKEN_ESTIMATOR_CONFIG.singleSpaceTokens),
    newlineTokens: numOr(t.newlineTokens, DEFAULT_TOKEN_ESTIMATOR_CONFIG.newlineTokens),
    messageOverhead: numOr(t.messageOverhead, DEFAULT_TOKEN_ESTIMATOR_CONFIG.messageOverhead),
    promptOverhead: numOr(t.promptOverhead, DEFAULT_TOKEN_ESTIMATOR_CONFIG.promptOverhead),
  };
}

function numOr(value: unknown, fallback: number): number {
  const n = typeof value === 'number' ? value : parseFloat(String(value));
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

/** Config-backed convenience wrappers used by the controller. */
export function estimatePromptTokens(formattedPrompt: string): number {
  return countPromptTokens(formattedPrompt, getEstimatorConfig());
}

export function estimateTextTokens(text: string): number {
  return countTokens(text, getEstimatorConfig());
}
