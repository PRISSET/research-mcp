import { TokenUsage } from './openai.types';

export interface ExtractedToolCall {
  id: string;
  name: string;
  arguments: string;
}

/**
 * Streaming state of the escaped-fence normalizer for a single buffer.
 *
 * `pending` holds a tail that could still turn into an escaped JSON fence, so
 * it is withheld from the buffer until the next chunk disambiguates it.
 *
 * The remaining fields track the JSON structure of the block body: a fence that
 * appears inside a string value is content, not a terminator, and only a fence
 * that follows a complete value ends the block.
 */
export interface FenceUnescapeState {
  pending: string;
  /** True while an escaped json fence is open. */
  inBlock: boolean;
  /** True once the JSON value inside the block has started. */
  inValue: boolean;
  /** Brace/bracket nesting depth of that JSON value. */
  depth: number;
  /** True while inside a JSON string literal. */
  inString: boolean;
  /** True when the previous character was an escape backslash in a string. */
  inStringEscape: boolean;
}

export interface ParserState {
  currentFragmentType: 'RESPONSE' | 'THINK' | 'REASONING';
  reasoningBuffer: string;
  responseBuffer: string;
  hasReasoning: boolean;
  lastReasoningSent: number;
  lastResponseSent: number;
  finalUsage: TokenUsage | null;
  title?: string;
  reasoningFence: FenceUnescapeState;
  responseFence: FenceUnescapeState;
}
