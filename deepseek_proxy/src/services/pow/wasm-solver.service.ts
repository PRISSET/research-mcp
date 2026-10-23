import fs from 'fs';
import { config } from '../../config';
import { solveDeepSeekHashV1 } from './js-solver.service';

let wasmInstance: WebAssembly.Instance | null = null;

export async function getWasmInstance(): Promise<WebAssembly.Instance | null> {
  if (wasmInstance) return wasmInstance;
  if (!fs.existsSync(config.wasmPath)) {
    return null;
  }
  try {
    const wasmBuffer = fs.readFileSync(config.wasmPath);
    const { instance } = await WebAssembly.instantiate(wasmBuffer);
    wasmInstance = instance;
    return wasmInstance;
  } catch (e) {
    return null;
  }
}

export function solvePowWasm(
  instance: WebAssembly.Instance | null,
  challenge: string,
  prefix: string,
  difficulty: number
): number {
  if (!instance) {
    return solveDeepSeekHashV1(challenge, prefix, difficulty);
  }

  try {
    const exports = instance.exports as any;
    const memory = exports.memory as WebAssembly.Memory;
    const encoder = new TextEncoder();

    function passString(str: string) {
      const encoded = encoder.encode(str);
      const ptr = exports.__wbindgen_export_0(encoded.length, 1);
      new Uint8Array(memory.buffer).set(encoded, ptr);
      return { ptr, len: encoded.length };
    }

    const s1 = passString(challenge);
    const s2 = passString(prefix);
    const sp = exports.__wbindgen_add_to_stack_pointer(-16);
    exports.wasm_solve(sp, s1.ptr, s1.len, s2.ptr, s2.len, difficulty);
    const isSome = new Int32Array(memory.buffer, sp, 1)[0];
    const answer = new Float64Array(memory.buffer, sp + 8, 1)[0];

    if (typeof exports.__wbindgen_add_to_stack_pointer === 'function') {
      exports.__wbindgen_add_to_stack_pointer(16);
    }

    if (!isSome) {
      return solveDeepSeekHashV1(challenge, prefix, difficulty);
    }
    return answer;
  } catch (err) {
    return solveDeepSeekHashV1(challenge, prefix, difficulty);
  }
}
