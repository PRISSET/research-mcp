import crypto from 'crypto';
import { config } from '../../config';

function getHasher() {
  try {
    const test = crypto.createHash('sha3-256');
    test.update('test');
    test.digest();
    return (data: string) => crypto.createHash('sha3-256').update(data).digest();
  } catch (_) {
    return (data: string) => crypto.createHash('sha256').update(data).digest();
  }
}

const hasher = getHasher();

export function solveDeepSeekHashV1(
  challenge: string,
  prefix: string,
  difficulty: number,
  maxAttempts: number = config.pow.jsFallbackLimit
): number {
  const fullBytes = Math.floor(difficulty / 8);
  const remainingBits = difficulty % 8;
  const mask = remainingBits === 0 ? 0 : (0xff << (8 - remainingBits)) & 0xff;

  for (let i = 0; i < maxAttempts; i++) {
    const input = prefix + i;
    const hash = hasher(input);
    let valid = true;

    for (let b = 0; b < fullBytes; b++) {
      if (hash[b] !== 0) {
        valid = false;
        break;
      }
    }

    if (valid && fullBytes < hash.length) {
      const byte = hash[fullBytes];
      if ((byte & mask) !== 0) {
        valid = false;
      }
    }

    if (valid) {
      return i;
    }
  }
  return -1;
}
