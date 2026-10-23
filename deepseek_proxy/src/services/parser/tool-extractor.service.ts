import crypto from 'crypto';
import { OpenAITool } from '../../types/openai.types';
import { ExtractedToolCall } from '../../types/common.types';

export interface ExtractedResult {
  toolCalls: ExtractedToolCall[];
  content: string;
  cleanReasoning: string;
}

/** Один обратный слэш, собранный без escape-последовательностей в исходнике. */
const BACKSLASH = String.fromCharCode(92);

/** Один символ бэктика. */
const TICK = String.fromCharCode(96);

/** Обычный закрывающий fence, то есть форма, которую понимает Harness. */
const PLAIN_CLOSE = TICK + TICK + TICK;

/** Открывающий fence с необязательным языковым тегом. */
const OPEN_FENCE_RE = new RegExp(TICK + TICK + TICK + '[a-zA-Z0-9_:-]*', 'g');

/** Пробельный символ, который может отделять fence от тела блока. */
function isWhitespace(ch: string): boolean {
  return ch === ' ' || ch === '\t' || ch === '\r' || ch === '\n';
}

/**
 * Ищет закрывающий fence блока, тело которого начинается с позиции `from`.
 *
 * Тело блока — JSON-значение, поэтому сканер учитывает строковые литералы:
 * fence, встретившийся внутри строки (частый случай при записи markdown-файла),
 * закрывашкой не считается. Возвращает индекс закрывающего fence или -1.
 */
function findJsonBlockEnd(source: string, from: number): number {
  const n = source.length;
  let i = from;

  while (i < n && isWhitespace(source[i])) i += 1;
  if (i >= n) return -1;

  const first = source[i];
  if (first === '{' || first === '[') {
    let depth = 0;
    let inString = false;
    let escaped = false;
    let closed = false;

    for (; i < n; i += 1) {
      const ch = source[i];
      if (inString) {
        if (escaped) escaped = false;
        else if (ch === BACKSLASH) escaped = true;
        else if (ch === '"') inString = false;
        continue;
      }
      if (ch === '"') {
        inString = true;
        continue;
      }
      if (ch === '{' || ch === '[') {
        depth += 1;
        continue;
      }
      if (ch === '}' || ch === ']') {
        depth -= 1;
        if (depth === 0) {
          i += 1;
          closed = true;
          break;
        }
      }
    }

    if (!closed) return -1;
  } else {
    // Скалярное значение: тело заканчивается вместе со строкой.
    while (i < n && source[i] !== '\n') i += 1;
  }

  while (i < n && isWhitespace(source[i])) i += 1;
  return source.startsWith(PLAIN_CLOSE, i) ? i : -1;
}

export function extractToolCalls(
  text: string = '',
  availableTools: OpenAITool[] = [],
  reasoningText: string = ''
): ExtractedResult {
  const validNames = new Set<string>();
  if (Array.isArray(availableTools) && availableTools.length > 0) {
    for (const t of availableTools) {
      const name = t?.function?.name || t?.name;
      if (name) validNames.add(name);
    }
  }

  function isValidToolCall(parsed: any): boolean {
    if (!parsed || typeof parsed !== 'object') return false;
    if (typeof parsed.name !== 'string' || !parsed.name.trim()) return false;
    if (validNames.size > 0 && !validNames.has(parsed.name)) return false;
    return Boolean(
      parsed.arguments !== undefined || parsed.parameters !== undefined
    );
  }

  function extractFromSource(sourceText: string) {
    const calls: Array<{ name: string; arguments: string; matchedBlock: string }> = [];
    let cleanText = sourceText || '';
    if (!sourceText || typeof sourceText !== 'string') {
      return { calls, cleanText: '' };
    }

    // 1. Поиск в markdown блоках кода.
    //    Границы блока определяются разбором JSON-структуры, а не нежадной
    //    регуляркой: fence внутри строкового значения обрывал бы такой матч и
    //    ломал извлечение tool call.
    OPEN_FENCE_RE.lastIndex = 0;
    let match: RegExpExecArray | null;
    while ((match = OPEN_FENCE_RE.exec(sourceText)) !== null) {
      const bodyStart = match.index + match[0].length;
      const closeIndex = findJsonBlockEnd(sourceText, bodyStart);
      if (closeIndex === -1) continue;

      const blockEnd = closeIndex + PLAIN_CLOSE.length;
      const matchedBlock = sourceText.slice(match.index, blockEnd);
      OPEN_FENCE_RE.lastIndex = blockEnd;

      try {
        const parsed = JSON.parse(sourceText.slice(bodyStart, closeIndex).trim());
        const items = Array.isArray(parsed) ? parsed : [parsed];
        let hasValid = false;

        for (const item of items) {
          if (isValidToolCall(item)) {
            const rawArgs =
              item.arguments !== undefined ? item.arguments : item.parameters;
            const args =
              typeof rawArgs === 'string'
                ? rawArgs
                : JSON.stringify(rawArgs || {});
            calls.push({
              name: item.name,
              arguments: args,
              matchedBlock,
            });
            hasValid = true;
          }
        }
        if (hasValid) {
          cleanText = cleanText.replace(matchedBlock, '');
        }
      } catch (e) {}
    }

    // 2. Поиск в XML-подобных тегах
    const xmlRegex =
      /<(?:tool_call|function_call|invoke)>\s*([\s\S]*?)\s*<\/(?:tool_call|function_call|invoke)>/g;
    while ((match = xmlRegex.exec(sourceText)) !== null) {
      try {
        const parsed = JSON.parse(match[1].trim());
        const items = Array.isArray(parsed) ? parsed : [parsed];
        let hasValid = false;

        for (const item of items) {
          if (isValidToolCall(item)) {
            const rawArgs =
              item.arguments !== undefined ? item.arguments : item.parameters;
            const args =
              typeof rawArgs === 'string'
                ? rawArgs
                : JSON.stringify(rawArgs || {});
            calls.push({
              name: item.name,
              arguments: args,
              matchedBlock: match[0],
            });
            hasValid = true;
          }
        }
        if (hasValid) {
          cleanText = cleanText.replace(match[0], '');
        }
      } catch (e) {}
    }

    // 3. Фолбэк: поиск голых JSON-объектов
    if (calls.length === 0) {
      const rawRegex =
        /\{\s*"name"\s*:\s*"([a-zA-Z0-9_\-]+)"\s*,\s*"(?:arguments|parameters)"\s*:\s*(\{[\s\S]*?\})\s*\}/g;
      while ((match = rawRegex.exec(sourceText)) !== null) {
        try {
          const toolName = match[1];
          if (validNames.size > 0 && !validNames.has(toolName)) continue;

          const rawArgs = JSON.parse(match[2]);
          calls.push({
            name: toolName,
            arguments: JSON.stringify(rawArgs),
            matchedBlock: match[0],
          });
          cleanText = cleanText.replace(match[0], '');
        } catch (e) {}
      }
    }

    return {
      calls,
      cleanText: cleanText.replace(/FINISHED$/, '').trim(),
    };
  }

  const fromResponse = extractFromSource(text);
  const fromReasoning = extractFromSource(reasoningText);

  const toolCalls: ExtractedToolCall[] = [];
  const seenCalls = new Set<string>();

  function registerCall(call: { name: string; arguments: string }) {
    const signature = `${call.name}::${call.arguments}`;
    if (!seenCalls.has(signature)) {
      seenCalls.add(signature);
      toolCalls.push({
        id: `call_${crypto.randomUUID().replace(/-/g, '').slice(0, 16)}`,
        name: call.name,
        arguments: call.arguments,
      });
    }
  }

  for (const c of fromResponse.calls) registerCall(c);
  for (const c of fromReasoning.calls) registerCall(c);

  return {
    toolCalls,
    content: fromResponse.cleanText,
    cleanReasoning: fromReasoning.cleanText,
  };
}
