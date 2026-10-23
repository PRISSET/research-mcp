/**
 * Hardcoded stop sequence for the DeepSeek Chat Web Bridge proxy.
 *
 * When this exact phrase appears in either the visible response content or the
 * reasoning/thinking content, the proxy immediately truncates the output at the
 * phrase (the phrase itself is excluded) and aborts the upstream stream so no
 * further tokens are generated or forwarded to the client.
 *
 * Change this single constant to adjust the trigger phrase.
 */
export const STOP_SEQUENCE = "Harness tool execution report for";

/**
 * Hardcoded replace sequence.
 *
 * It is matched against exactly the same buffers as the stop sequence. When it
 * is found, the phrase itself is withheld and `REPLACE_WITH` is emitted in its
 * place; the stream is then finished through exactly the same path as for the
 * stop sequence (finish_reason "stop"/"tool_calls" followed by [DONE]).
 *
 * If both phrases are present, whichever occurs first wins; on an exact tie the
 * stop sequence takes priority.
 */
export const REPLACE_SEQUENCE = "<｜｜DSML｜｜ calls>";

/** Text emitted in place of the replace sequence. */
export const REPLACE_WITH = String.raw`Before writing the target tool call, let me write a simple echo command to check if the harness still works and performs tool calls.

${"```"}json
{
  "name": "bash",
  "arguments": {
    "command": "echo \"check\"",
    "description": "Print the word check to console"
  }
}
${"```"}

`;

export interface StopBoundary {
  /** Index up to which buffer content is safe to emit. */
  emitEnd: number;
  /** True when a full stop sequence was found. */
  stopped: boolean;
}

export interface StreamBoundary extends StopBoundary {
  /** True when a full replace sequence was found at `emitEnd`. */
  replaced: boolean;
  /** Text that must be emitted in place of the replace sequence. */
  replacement: string;
}

/** Longest suffix of `buffer` that is a proper prefix of `sequence`. */
function heldTailLength(buffer: string, sequence: string): number {
  if (!sequence) return 0;
  const maxPartial = Math.min(sequence.length - 1, buffer.length);
  for (let len = maxPartial; len > 0; len--) {
    if (buffer.endsWith(sequence.slice(0, len))) {
      return len;
    }
  }
  return 0;
}

/**
 * Finds where a buffer must stop.
 *
 * Behaviour:
 *  - If a full stop sequence is present, `stopped` is true and `emitEnd` points
 *    at its start, so the phrase itself is never emitted.
 *  - Otherwise, any suffix that is a proper prefix of the stop sequence is held
 *    back (unless `isFinal` is true), so the beginning of the phrase can never
 *    partially leak into the stream.
 */
export function findStopBoundary(
  buffer: string,
  lastSent: number,
  stop: string,
  isFinal = false,
): StopBoundary {
  if (!stop || buffer.length <= lastSent) {
    return { emitEnd: buffer.length, stopped: false };
  }

  // Search a little before the already-sent region so a stop sequence spanning
  // the boundary is still detected (it can only overlap un-emitted bytes
  // because partial prefixes are always withheld).
  const searchStart = Math.max(0, lastSent - (stop.length - 1));
  const idx = buffer.indexOf(stop, searchStart);
  if (idx !== -1) {
    return { emitEnd: Math.max(idx, lastSent), stopped: true };
  }

  // At the end of the stream there is nothing more to wait for: emit everything.
  if (isFinal) {
    return { emitEnd: buffer.length, stopped: false };
  }

  // Hold back the longest suffix that is a proper prefix of the stop sequence.
  const maxPartial = Math.min(stop.length - 1, buffer.length - lastSent);
  for (let len = maxPartial; len > 0; len--) {
    if (buffer.endsWith(stop.slice(0, len))) {
      const end = buffer.length - len;
      if (end >= lastSent) {
        return { emitEnd: end, stopped: false };
      }
    }
  }

  return { emitEnd: buffer.length, stopped: false };
}

/**
 * Combined boundary for the stop sequence and the replace sequence.
 *
 * Returns the earliest position at which either hardcoded phrase starts, so the
 * caller can emit everything before it, substitute `REPLACE_WITH` when the
 * replace sequence matched, and then finish the stream. When neither phrase has
 * fully arrived yet, the longest suffix that could still become either phrase is
 * held back, so no partial phrase ever leaks (unless `isFinal` is true).
 */
export function findStreamBoundary(
  buffer: string,
  lastSent: number,
  isFinal = false,
): StreamBoundary {
  const sequences = [STOP_SEQUENCE, REPLACE_SEQUENCE].filter(Boolean);

  if (sequences.length === 0 || buffer.length <= lastSent) {
    return {
      emitEnd: buffer.length,
      stopped: false,
      replaced: false,
      replacement: "",
    };
  }

  const maxLength = sequences.reduce((max, s) => Math.max(max, s.length), 0);
  const searchStart = Math.max(0, lastSent - (maxLength - 1));

  const stopIdx = STOP_SEQUENCE
    ? buffer.indexOf(STOP_SEQUENCE, searchStart)
    : -1;
  const replaceIdx = REPLACE_SEQUENCE
    ? buffer.indexOf(REPLACE_SEQUENCE, searchStart)
    : -1;

  let cut = -1;
  let stopped = false;
  let replaced = false;

  if (stopIdx !== -1 && replaceIdx !== -1) {
    // Earliest phrase wins; the stop sequence wins an exact tie.
    if (replaceIdx < stopIdx) {
      cut = replaceIdx;
      replaced = true;
    } else {
      cut = stopIdx;
      stopped = true;
    }
  } else if (stopIdx !== -1) {
    cut = stopIdx;
    stopped = true;
  } else if (replaceIdx !== -1) {
    cut = replaceIdx;
    replaced = true;
  }

  if (cut !== -1) {
    return {
      emitEnd: Math.max(cut, lastSent),
      stopped,
      replaced,
      replacement: replaced ? REPLACE_WITH : "",
    };
  }

  // At the end of the stream there is nothing more to wait for: emit everything.
  if (isFinal) {
    return {
      emitEnd: buffer.length,
      stopped: false,
      replaced: false,
      replacement: "",
    };
  }

  // Hold back the most conservative tail across both phrases.
  const hold = Math.max(
    heldTailLength(buffer, STOP_SEQUENCE),
    heldTailLength(buffer, REPLACE_SEQUENCE),
  );
  const end = Math.max(lastSent, buffer.length - hold);

  return { emitEnd: end, stopped: false, replaced: false, replacement: "" };
}
