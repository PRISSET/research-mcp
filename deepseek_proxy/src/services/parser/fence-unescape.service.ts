import { FenceUnescapeState } from '../../types/common.types';

/** A single backslash. */
const BACKSLASH = String.fromCharCode(92);

/** A single backtick. */
const TICK = '`';

/** One escaped backtick (\`). */
const ESCAPED_TICK = BACKSLASH + TICK;

/** Escaped opening fence: \`\`\`json */
const ESCAPED_OPEN = ESCAPED_TICK + ESCAPED_TICK + ESCAPED_TICK + 'json';

/** Escaped closing fence: \`\`\` */
const ESCAPED_CLOSE = ESCAPED_TICK + ESCAPED_TICK + ESCAPED_TICK;

/** Plain closing fence. */
const PLAIN_CLOSE = TICK + TICK + TICK;

/** Plain opening fence. */
const PLAIN_OPEN = TICK + TICK + TICK + 'json';

export function createFenceUnescapeState(): FenceUnescapeState {
  return {
    pending: '',
    inBlock: false,
    inValue: false,
    depth: 0,
    inString: false,
    inStringEscape: false,
  };
}

/**
 * Checks if the remainder of `buf` starting at `i` is a non-empty proper prefix of `sequence`.
 * O(1) allocation because length of sequence <= 10.
 */
function isProperPrefixOf(buf: string, i: number, sequence: string): boolean {
  const rem = buf.length - i;
  return rem > 0 && rem < sequence.length && sequence.startsWith(buf.slice(i));
}

/**
 * Updates JSON bracket and string state.
 */
function updateJsonState(state: FenceUnescapeState, ch: string): void {
  if (state.inString) {
    if (state.inStringEscape) {
      state.inStringEscape = false;
    } else if (ch === BACKSLASH) {
      state.inStringEscape = true;
    } else if (ch === '"') {
      state.inString = false;
    }
    return;
  }

  if (ch === '"') {
    state.inString = true;
    state.inValue = true;
    return;
  }
  if (ch === '{' || ch === '[') {
    state.depth += 1;
    state.inValue = true;
    return;
  }
  if (ch === '}' || ch === ']') {
    if (state.depth > 0) state.depth -= 1;
    return;
  }
  if (ch.trim() !== '') {
    state.inValue = true;
  }
}

/**
 * Streaming-safe transformer for escaped fences.
 * Optimized for token-by-token processing.
 */
export function pushFenceChunk(state: FenceUnescapeState, chunk: string): string {
  if (!chunk && !state.pending) return '';

  const buf = state.pending + chunk;
  state.pending = '';

  let out = '';
  let i = 0;
  const n = buf.length;

  while (i < n) {
    if (state.inBlock) {
      // Check for fences only when not inside a JSON string
      if (!state.inString) {
        // A. Handle potential duplicate opening fence (only valid before JSON value starts)
        if (!state.inValue && state.depth === 0) {
          if (buf.startsWith(ESCAPED_OPEN, i)) {
            // Drop duplicate opener
            i += ESCAPED_OPEN.length;
            continue;
          }
          // If buffer ends with an ambiguous prefix of ESCAPED_OPEN (e.g. \`\`\`j)
          if (isProperPrefixOf(buf, i, ESCAPED_OPEN)) {
            state.pending = buf.slice(i);
            break;
          }
        }

        // B. Handle closing fences (valid both after JSON value and on empty blocks when depth === 0)
        if (state.depth === 0) {
          if (buf.startsWith(ESCAPED_CLOSE, i)) {
            out += PLAIN_CLOSE;
            state.inBlock = false;
            state.inValue = false;
            i += ESCAPED_CLOSE.length;
            continue;
          }
          if (buf.startsWith(PLAIN_CLOSE, i)) {
            out += PLAIN_CLOSE;
            state.inBlock = false;
            state.inValue = false;
            i += PLAIN_CLOSE.length;
            continue;
          }
          // Buffer tail only if it could become a closing fence
          if (isProperPrefixOf(buf, i, ESCAPED_CLOSE) || isProperPrefixOf(buf, i, PLAIN_CLOSE)) {
            state.pending = buf.slice(i);
            break;
          }
        }
      }

      // C. Inside block: replace escaped ticks (\` -> `)
      if (buf.startsWith(ESCAPED_TICK, i)) {
        out += TICK;
        updateJsonState(state, TICK);
        i += ESCAPED_TICK.length;
        continue;
      }

      // D. Withhold a single trailing backslash at the very end of the chunk
      if (i === n - 1 && buf[i] === BACKSLASH) {
        state.pending = BACKSLASH;
        break;
      }

      // E. Normal character inside block
      const ch = buf[i];
      out += ch;
      updateJsonState(state, ch);
      i += 1;
      continue;
    }

    // --- OUTSIDE BLOCK ---
    if (buf.startsWith(ESCAPED_OPEN, i)) {
      out += PLAIN_OPEN;
      state.inBlock = true;
      state.inValue = false;
      state.depth = 0;
      state.inString = false;
      state.inStringEscape = false;
      i += ESCAPED_OPEN.length;
      continue;
    }

    if (isProperPrefixOf(buf, i, ESCAPED_OPEN)) {
      state.pending = buf.slice(i);
      break;
    }

    out += buf[i];
    i += 1;
  }

  return out;
}

/**
 * Flushes whatever is held back at the end of the stream.
 * Automatically resolves any unfinished closing fence into standard format.
 */
export function flushFenceChunk(state: FenceUnescapeState): string {
  let pending = state.pending;
  state.pending = '';

  if (!pending) return '';

  if (state.inBlock) {
    // If stream ended right on a closing fence
    if (pending === ESCAPED_CLOSE || pending.endsWith(ESCAPED_CLOSE)) {
      state.inBlock = false;
      state.inValue = false;
      const prefix = pending.slice(0, -ESCAPED_CLOSE.length);
      return prefix + PLAIN_CLOSE;
    }
    // Convert any remaining escaped backticks
    if (pending.includes(ESCAPED_TICK)) {
      pending = pending.split(ESCAPED_TICK).join(TICK);
    }
  }

  return pending;
}

/**
 * Helper Async Generator to safely wrap an incoming LLM token stream.
 * Guarantees that `flushFenceChunk` is called even if stream errors or breaks early.
 */
export async function* unescapeFenceStream(
  inputStream: AsyncIterable<string>
): AsyncGenerator<string, void, unknown> {
  const state = createFenceUnescapeState();
  try {
    for await (const chunk of inputStream) {
      if (!chunk) continue;
      const out = pushFenceChunk(state, chunk);
      if (out) yield out;
    }
  } finally {
    const tail = flushFenceChunk(state);
    if (tail) yield tail;
  }
}
