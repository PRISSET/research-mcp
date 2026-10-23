export interface DeepSeekChallengeResponse {
  code?: number;
  msg?: string;
  data?: {
    biz_code?: number;
    biz_msg?: string;
    biz_data?: {
      challenge: DeepSeekChallenge;
    };
  };
}

export interface DeepSeekChallenge {
  algorithm: string;
  challenge: string;
  salt: string;
  difficulty: number;
  expire_at: number;
  signature: string;
  target_path: string;
}

export interface DeepSeekPowAnswerPayload {
  algorithm: string;
  challenge: string;
  salt: string;
  answer: number;
  signature: string;
  target_path: string;
}

export interface DeepSeekChatPayload {
  chat_session_id: string;
  parent_message_id: string | null;
  prompt: string;
  ref_file_ids: string[];
  thinking_enabled: boolean;
  search_enabled: boolean;
  model_type: 'default' | string;
  action: null;
  preempt: boolean;
}

export type FragmentType = 'RESPONSE' | 'THINK' | 'REASONING';

export interface DeepSeekSSEData {
  p?: string;
  o?: 'BATCH' | 'APPEND' | string;
  v?: any;
}
