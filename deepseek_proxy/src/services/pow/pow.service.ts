import https from 'https';
import { config } from '../../config';
import { DeepSeekChallengeResponse, DeepSeekPowAnswerPayload } from '../../types/deepseek.types';
import { getWasmInstance, solvePowWasm } from './wasm-solver.service';

export async function getPowResponseHeader(userToken: string, targetPath: string): Promise<string> {
  const challengeRes = await new Promise<DeepSeekChallengeResponse>((resolve, reject) => {
    const req = https.request(
      {
        hostname: config.targetHost,
        port: 443,
        path: '/api/v0/chat/create_pow_challenge',
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
            const err: any = new Error(`PoW challenge HTTP ${status}: ${body}`);
            err.status = status;
            reject(err);
            return;
          }
          try {
            resolve(JSON.parse(body));
          } catch (e) {
            reject(new Error(`Failed to parse PoW challenge response: ${body}`));
          }
        });
      }
    );
    req.on('error', reject);
    req.write(JSON.stringify({ target_path: targetPath }));
    req.end();
  });

  const challengeObj = challengeRes.data?.biz_data?.challenge;
  if (!challengeObj) {
    throw new Error(`Invalid challenge response: ${JSON.stringify(challengeRes)}`);
  }

  const { algorithm, challenge, salt, difficulty, expire_at, signature } = challengeObj;
  const prefix = `${salt}_${expire_at}_`;
  const wasm = await getWasmInstance();
  const answer = solvePowWasm(wasm, challenge, prefix, difficulty);

  const answerPayload: DeepSeekPowAnswerPayload = {
    algorithm,
    challenge,
    salt,
    answer,
    signature,
    target_path: targetPath,
  };
  return Buffer.from(JSON.stringify(answerPayload)).toString('base64');
}
