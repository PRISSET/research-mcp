import { DeepSeekSSEData, FragmentType } from '../../types/deepseek.types';
import { ParserState } from '../../types/common.types';
import {
  createFenceUnescapeState,
  pushFenceChunk,
  flushFenceChunk,
} from './fence-unescape.service';

export function createParserState(): ParserState {
  return {
    currentFragmentType: 'RESPONSE',
    reasoningBuffer: '',
    responseBuffer: '',
    hasReasoning: false,
    lastReasoningSent: 0,
    lastResponseSent: 0,
    finalUsage: null,
    reasoningFence: createFenceUnescapeState(),
    responseFence: createFenceUnescapeState(),
  };
}

export function processDeepSeekSSEData(data: DeepSeekSSEData, state: ParserState, eventName: string | null): void {
  if (!state.reasoningBuffer) state.reasoningBuffer = '';
  if (!state.responseBuffer) state.responseBuffer = '';
  if (state.hasReasoning === undefined) state.hasReasoning = false;
  if (!state.currentFragmentType) state.currentFragmentType = 'RESPONSE';
  if (!state.reasoningFence) state.reasoningFence = createFenceUnescapeState();
  if (!state.responseFence) state.responseFence = createFenceUnescapeState();

  if (
    eventName === 'title' ||
    eventName === 'update_session' ||
    eventName === 'ready' ||
    eventName === 'close' ||
    eventName === 'finish' ||
    eventName === 'ping'
  ) {
    if (eventName === 'title' && data && typeof (data as any).content === 'string') {
      state.title = (data as any).content;
    }
    return;
  }

  if (!data || typeof data !== 'object') return;

  function appendContent(content: string, fragType?: FragmentType) {
    if (!content || typeof content !== 'string') return;
    const type = fragType || state.currentFragmentType || 'RESPONSE';
    if (type === 'THINK' || type === 'REASONING') {
      state.reasoningBuffer += pushFenceChunk(state.reasoningFence, content);
      state.hasReasoning = true;
    } else {
      state.responseBuffer += pushFenceChunk(state.responseFence, content);
    }
  }

  // 1. Инициализирующий пакет ответа
  if (data.v && typeof data.v === 'object' && data.v.response) {
    const resp = data.v.response;
    if (Array.isArray(resp.fragments) && resp.fragments.length > 0) {
      for (const frag of resp.fragments) {
        if (frag && typeof frag === 'object') {
          state.currentFragmentType = frag.type || 'RESPONSE';
          if (frag.content) {
            appendContent(frag.content, frag.type);
          }
        }
      }
    }
    if (typeof resp.accumulated_token_usage === 'number') {
      state.finalUsage = {
        prompt_tokens: 0,
        completion_tokens: resp.accumulated_token_usage,
        total_tokens: resp.accumulated_token_usage,
      };
    }
    return;
  }

  // 2. Batch-события
  if (data.o === 'BATCH' && Array.isArray(data.v)) {
    for (const item of data.v) {
      if (!item || typeof item !== 'object') continue;
      const itemP = item.p || '';
      if (
        itemP === 'accumulated_token_usage' ||
        itemP === 'response/accumulated_token_usage'
      ) {
        if (typeof item.v === 'number') {
          state.finalUsage = { prompt_tokens: 0, completion_tokens: item.v, total_tokens: item.v };
        }
      } else if (
        (itemP === 'fragments' || itemP === 'response/fragments') &&
        item.o === 'APPEND' &&
        Array.isArray(item.v)
      ) {
        for (const frag of item.v) {
          if (frag && typeof frag === 'object') {
            state.currentFragmentType = frag.type || 'RESPONSE';
            if (frag.content) {
              appendContent(frag.content, frag.type);
            }
          }
        }
      } else if (
        typeof itemP === 'string' &&
        (itemP.endsWith('/content') || itemP === 'content') &&
        typeof item.v === 'string'
      ) {
        appendContent(item.v, state.currentFragmentType);
      }
    }
    return;
  }

  // 3. Добавление нового фрагмента
  if (
    data.p === 'response/fragments' &&
    data.o === 'APPEND' &&
    Array.isArray(data.v)
  ) {
    for (const frag of data.v) {
      if (frag && typeof frag === 'object') {
        state.currentFragmentType = frag.type || 'RESPONSE';
        if (frag.content) {
          appendContent(frag.content, frag.type);
        }
      }
    }
    return;
  }

  // 4. Прямое обновление токенов
  if (
    (data.p === 'response/accumulated_token_usage' ||
      data.p === 'accumulated_token_usage') &&
    typeof data.v === 'number'
  ) {
    state.finalUsage = { prompt_tokens: 0, completion_tokens: data.v, total_tokens: data.v };
    return;
  }

  // 5. Инкремент текста по JSON pointer
  if (
    typeof data.p === 'string' &&
    (data.p.endsWith('/content') || data.p === 'content')
  ) {
    if (typeof data.v === 'string') {
      appendContent(data.v, state.currentFragmentType);
    }
    return;
  }

  // 6. Игнорируем служебные поля
  if (data.p !== undefined && data.p !== null) {
    return;
  }

  // 7. Обычные текстовые токены
  if (typeof data.v === 'string') {
    appendContent(data.v, state.currentFragmentType);
    return;
  }

  // 8. Резервная обработка вложенных фрагментов
  if (data.v && typeof data.v === 'object') {
    if (typeof data.v.content === 'string') {
      appendContent(data.v.content, state.currentFragmentType);
      return;
    }
    if (Array.isArray(data.v.fragments)) {
      for (const frag of data.v.fragments) {
        if (frag && typeof frag === 'object') {
          state.currentFragmentType = frag.type || 'RESPONSE';
          if (frag.content) {
            appendContent(frag.content, frag.type);
          }
        }
      }
      return;
    }
  }
}

/**
 * Releases the tail held back by the fence normalizer at the end of a stream.
 *
 * The normalizer withholds a short suffix that could still have turned into an
 * escaped fence, so the buffers are only complete once the stream is over.
 * Must be called before the buffers are consumed (boundary checks, tool-call
 * extraction, usage estimation).
 */
export function flushFenceBuffers(state: ParserState): void {
  state.reasoningBuffer += flushFenceChunk(state.reasoningFence);
  state.responseBuffer += flushFenceChunk(state.responseFence);
}
