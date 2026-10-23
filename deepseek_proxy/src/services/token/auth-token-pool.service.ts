/**
 * Authorization-token pool.
 *
 * The harness (or any OpenAI-compatible client) may supply a pipe-delimited
 * list of DeepSeek web tokens as the bearer value:
 *
 *     Authorization: Bearer token1|token2|token3
 *
 * Every request is served by exactly one token from that list, chosen
 * round-robin so no single token absorbs the whole request rate. A token that
 * answers with a throttle status (429/503/529) is put in exponential cooldown
 * and skipped until the cooldown expires, which keeps the proxy off the
 * per-token rate limit as long as at least one token is healthy.
 *
 * The pool is intentionally process-global and stateful: rotation only means
 * something across requests. State is keyed by the token string, so a list
 * that changes between requests simply gains new entries.
 */
import { config } from '../../config';

interface TokenState {
  /** Epoch ms until which this token must be skipped. 0 = healthy. */
  cooledUntil: number;
  /** Consecutive throttle strikes; drives the exponential backoff. */
  strikes: number;
  /** Epoch ms of the last time this token was handed out. */
  lastUsed: number;
}

const states = new Map<string, TokenState>();

/** Monotonic round-robin cursor shared by all lists. */
let rrCounter = 0;

function getState(token: string): TokenState {
  let state = states.get(token);
  if (!state) {
    state = { cooledUntil: 0, strikes: 0, lastUsed: 0 };
    states.set(token, state);
  }
  return state;
}

/**
 * Split a raw bearer value into distinct, non-empty tokens.
 * A value without `|` yields a single-element list, so behaviour for a single
 * token is unchanged.
 */
export function parseTokenList(raw: string | undefined | null): string[] {
  if (!raw) return [];
  const seen = new Set<string>();
  const out: string[] = [];
  for (const part of raw.split('|')) {
    const token = part.trim();
    if (token && !seen.has(token)) {
      seen.add(token);
      out.push(token);
    }
  }
  return out;
}

function pickFrom(pool: string[]): string {
  const token = pool[rrCounter % pool.length];
  rrCounter = (rrCounter + 1) % 1000000007;
  getState(token).lastUsed = Date.now();
  return token;
}

/**
 * Resolve the next usable token from a raw bearer value.
 * Healthy tokens are preferred; if every token is cooling down the list is
 * still rotated over rather than failing the request outright.
 */
export function resolveToken(raw: string): string {
  const list = parseTokenList(raw);
  if (list.length === 0) return '';
  if (!config.authToken.rotationEnabled || list.length === 1) {
    getState(list[0]).lastUsed = Date.now();
    return list[0];
  }
  const now = Date.now();
  const healthy = list.filter((t) => getState(t).cooledUntil <= now);
  return pickFrom(healthy.length > 0 ? healthy : list);
}

/**
 * Resolve a token different from `exclude`, used for in-request failover when
 * the current token has just been throttled.
 */
export function resolveTokenExcluding(raw: string, exclude: string): string {
  const list = parseTokenList(raw);
  const others = list.filter((t) => t !== exclude);
  if (others.length === 0) return list[0] || exclude || '';
  if (others.length === 1) {
    getState(others[0]).lastUsed = Date.now();
    return others[0];
  }
  const now = Date.now();
  const healthy = others.filter((t) => getState(t).cooledUntil <= now);
  return pickFrom(healthy.length > 0 ? healthy : others);
}

/**
 * Record a throttle response for `token`. The cooldown grows exponentially with
 * consecutive strikes and is capped by `TOKEN_MAX_COOLDOWN_MS`. An explicit
 * `retryAfterMs` (parsed from the upstream error when available) is used as the
 * base instead of the default.
 */
export function reportRateLimited(token: string, retryAfterMs?: number): void {
  if (!token) return;
  const state = getState(token);
  state.strikes += 1;
  const base =
    typeof retryAfterMs === 'number' && retryAfterMs > 0
      ? retryAfterMs
      : config.authToken.cooldownMs;
  const backoff = Math.min(base * Math.pow(2, state.strikes - 1), config.authToken.maxCooldownMs);
  state.cooledUntil = Date.now() + backoff;
}

/** Clear cooldown/strikes for a token after a successful upstream call. */
export function reportSuccess(token: string): void {
  if (!token) return;
  const state = getState(token);
  state.strikes = 0;
  state.cooledUntil = 0;
}

export interface TokenPoolInfo {
  /** Number of distinct tokens in the supplied list. */
  total: number;
  /** Tokens currently outside cooldown. */
  available: number;
  /** Tokens currently in cooldown. */
  cooled: number;
  /** True when the list actually contains more than one token. */
  rotation: boolean;
}

/** Snapshot of the pool for the health/status endpoint. */
export function getPoolInfo(raw: string): TokenPoolInfo {
  const list = parseTokenList(raw);
  const now = Date.now();
  const cooled = list.filter((t) => getState(t).cooledUntil > now).length;
  return {
    total: list.length,
    available: list.length - cooled,
    cooled,
    rotation: list.length > 1,
  };
}
