import fs from 'fs';
import path from 'path';
import yaml from 'js-yaml';

export interface ProviderOptions {
  providerId?: string;
  tokenRef?: string;
  baseURL?: string;
  model?: string;
  displayName?: string;
}

export function loadSettings(filePath: string): Record<string, any> {
  if (!fs.existsSync(filePath)) {
    return {};
  }
  try {
    const raw = fs.readFileSync(filePath, 'utf-8');
    return (yaml.load(raw) as Record<string, any>) || {};
  } catch (err: any) {
    console.warn(`Failed to load settings from ${filePath}:`, err.message);
    return {};
  }
}

export function saveSettings(filePath: string, settings: Record<string, any>): void {
  const dir = path.dirname(filePath);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
  fs.writeFileSync(filePath, yaml.dump(settings, { noRefs: true }), 'utf-8');
}

export function addProvider(settings: Record<string, any>, options: ProviderOptions): void {
  const {
    providerId = 'deepseek-chat-web',
    tokenRef = 'DEEPSEEK_CHAT_TOKEN',
    baseURL = 'http://127.0.0.1:3085',
    model = 'deepseek-chat',
    displayName = 'DeepSeek Chat (Web)',
  } = options;

  if (!settings.llm) settings.llm = {};
  if (!settings.llm['llm-pi-ai']) settings.llm['llm-pi-ai'] = {};
  const llmSection = settings.llm['llm-pi-ai'];
  if (!llmSection.providers) llmSection.providers = {};

  llmSection.providers[providerId] = {
    displayName,
    apiKeyEnv: tokenRef,
    baseURL,
    api: 'openai-completions',
    // Fallback request modalities for models this route lists that neither the
    // model entry nor the installed pi-ai catalog describes. `deepseek-chat-web`
    // is not a catalog provider, so this route-level declaration is what lets
    // Harness treat the proxy as vision-capable.
    defaultInput: ['text', 'image'],
    models: [
      {
        id: model,
        name: 'DeepSeek Chat',
        contextWindow: 1000000,
        maxTokens: 256000,
        // Declaring `image` is what makes a hand-declared vision model usable:
        // without it Harness rejects image input before the proxy is reached.
        input: ['text', 'image'],
      },
    ],
  };
}

export function setDefaultModel(settings: Record<string, any>, providerId: string, model: string): void {
  if (!settings.llm) settings.llm = {};
  settings.llm['agent-default-model'] = {
    provider: providerId,
    model: model,
  };
}

export function applyConfig(filePath: string, options: ProviderOptions): void {
  const settings = loadSettings(filePath);
  addProvider(settings, options);
  setDefaultModel(settings, options.providerId || 'deepseek-chat-web', options.model || 'deepseek-chat');
  saveSettings(filePath, settings);
}
