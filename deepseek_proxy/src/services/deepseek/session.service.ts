import https from 'https';
import { config } from '../../config';

const userSessions = new Map<string, string>();

export async function getOrCreateChatSession(userToken: string, agent: string = 'chat'): Promise<string> {
  if (userSessions.has(userToken)) {
    return userSessions.get(userToken)!;
  }
  const sid = await createChatSession(userToken, agent);
  if (sid) {
    userSessions.set(userToken, sid);
  }
  return sid;
}

function createChatSession(userToken: string, agent: string = 'chat'): Promise<string> {
  return new Promise((resolve, reject) => {
    const req = https.request(
      {
        hostname: config.targetHost,
        port: 443,
        path: '/api/v0/chat_session/create',
        method: 'POST',
        headers: {
          Authorization: `Bearer ${userToken}`,
          'Content-Type': 'application/json',
          ...config.clientHeaders,
        },
      },
      (res) => {
        let body = '';
        res.on('data', (c) => (body += c));
        res.on('end', () => {
          const status = res.statusCode || 0;
          // Surface the HTTP status on the error so the caller can tell a
          // throttled token (429/503) from a genuinely broken one and rotate.
          if (status !== 200) {
            const err: any = new Error(`Chat session HTTP ${status}: ${body}`);
            err.status = status;
            reject(err);
            return;
          }
          try {
            const json = JSON.parse(body);
            const sid = json.data?.biz_data?.chat_session?.id || json.data?.biz_data?.id;
            if (sid) resolve(sid);
            else reject(new Error(`Failed to extract chat_session_id from ${body}`));
          } catch (e) {
            reject(new Error(`Failed to create session: ${body}`));
          }
        });
      }
    );
    req.on('error', reject);
    req.write(JSON.stringify({ agent }));
    req.end();
  });
}
