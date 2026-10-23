#!/usr/bin/env node
import path from 'path';
import { applyConfig } from '../config/dsh-settings';

const settingsPath =
  process.env.DSCW_SETTINGS ||
  path.resolve(process.env.HOME || '/home/tvijas', '.dsh/settings.yaml');
const providerId = process.env.DSCW_PROV || 'deepseek-chat-web';
const tokenRef = process.env.DSCW_TOKEN_REF || 'DEEPSEEK_CHAT_TOKEN';
const baseURL = process.env.DSCW_PROXY_URL || 'http://127.0.0.1:3085';
const model = process.env.DSCW_MODEL || 'deepseek-chat';

try {
  applyConfig(settingsPath, { providerId, tokenRef, baseURL, model });
  console.log(`OK: provider ${providerId} added to ${settingsPath}`);
  console.log(`OK: default model set to ${providerId}/${model}`);
} catch (err: any) {
  console.error('FAIL:', err.message);
  process.exit(1);
}
