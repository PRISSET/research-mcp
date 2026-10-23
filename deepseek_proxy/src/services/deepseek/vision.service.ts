import fs from 'fs';
import path from 'path';
import https from 'https';
import http from 'http';
import { execSync } from 'child_process';
import { config } from '../../config';
import { getPowResponseHeader } from '../pow/pow.service';
import { OpenAIMessage } from '../../types/openai.types';

function downloadImage(url: string): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const protocol = url.startsWith('https') ? https : http;
    const req = protocol.get(url, (res) => {
      if (res.statusCode && res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        downloadImage(res.headers.location).then(resolve).catch(reject);
        return;
      }
      if (res.statusCode !== 200) {
        reject(new Error(`Download failed: ${res.statusCode}`));
        res.resume();
        return;
      }
      const chunks: Buffer[] = [];
      res.on('data', (c) => chunks.push(c));
      res.on('end', () => resolve(Buffer.concat(chunks)));
    });
    req.on('error', reject);
    req.setTimeout(30000, () => {
      req.destroy();
      reject(new Error('Image download timeout'));
    });
  });
}

/**
 * A resolved image source. Strings are data URLs (`data:image/...;base64,...`),
 * remote URLs, or local paths; Buffers are raw image bytes. Either form is
 * accepted by `uploadAndPrepareImage`, so the extraction layer never has to
 * touch the filesystem for inline attachments.
 */
export type ImageSource = string | Buffer;

let cachedSearchDirs: string[] | null = null;
function getSearchDirs(): string[] {
  if (!cachedSearchDirs) {
    cachedSearchDirs = [
      process.cwd(),
      path.join(process.env.HOME || '', 'projects'),
      path.join(process.env.HOME || '', 'projects/dhs_test'),
      path.join(process.env.HOME || '', 'projects/dhs_test/hello'),
    ].filter(Boolean);
  }
  return cachedSearchDirs;
}

/**
 * Normalize an OpenAI Responses `input_file.file_data` payload into an image
 * data URL. A non-image media type (`data:application/pdf;...`) yields null,
 * because the DeepSeek vision pipeline only accepts images.
 */
function toImageDataUrl(fileData: string): string | null {
  const trimmed = fileData.trim();
  if (trimmed.startsWith('data:')) {
    return /^data:image\//i.test(trimmed) ? trimmed : null;
  }
  // Bare base64 with no declared mime type: assume PNG.
  return trimmed ? `data:image/png;base64,${trimmed}` : null;
}

/** Harvest images referenced by a free-form text body (markdown, URLs, paths). */
function collectImagesFromText(content: string, images: ImageSource[]): void {
  const mdMatches = content.match(/!\[[^\]]*\]\(([^)]+)\)/g);
  if (mdMatches) {
    for (const match of mdMatches) {
      const url = match.match(/\(([^)]+)\)/)?.[1];
      if (url) images.push(url);
    }
  }
  const urlMatches = content.match(/https?:\/\/[^\s\)\"\'<>]+\.(?:png|jpg|jpeg|webp|gif)/gi);
  if (urlMatches) images.push(...urlMatches);

  const fileMatches = content.match(/(?:[\w\-\.\/\\~]+?\.(?:png|jpg|jpeg|webp|gif|bmp))/gi);
  if (fileMatches) {
    for (const match of fileMatches) {
      const cleanPath = match.replace(/^['"]|['"]$/g, '').trim();
      if (fs.existsSync(cleanPath)) {
        images.push(path.resolve(cleanPath));
        continue;
      }
      for (const dir of getSearchDirs()) {
        const candidate = path.resolve(dir, cleanPath);
        if (fs.existsSync(candidate)) {
          images.push(candidate);
          break;
        }
      }
    }
  }
}

/**
 * Recursively collect image sources from any content shape used by the
 * OpenAI chat-completions, OpenAI Responses, and Anthropic Messages APIs.
 */
function collectImageParts(content: unknown, images: ImageSource[], fromText: boolean = true): void {
  if (!content) return;

  if (Array.isArray(content)) {
    for (const part of content) collectImageParts(part, images, fromText);
    return;
  }

  if (typeof content === 'string') {
    if (fromText) collectImagesFromText(content, images);
    return;
  }

  if (typeof content !== 'object') return;
  const part: any = content;

  switch (part.type) {
    case 'image_url': {
      const url = typeof part.image_url === 'string' ? part.image_url : part.image_url?.url;
      if (typeof url === 'string' && url) images.push(url);
      return;
    }
    // OpenAI Responses: image travels as a data URL, remote URL, or file_id.
    case 'input_image': {
      const url = typeof part.image_url === 'string' ? part.image_url : part.image_url?.url;
      if (typeof url === 'string' && url) images.push(url);
      return;
    }
    // OpenAI Responses: base64 bytes, optionally wrapped as a data URL.
    case 'input_file': {
      if (typeof part.file_data === 'string' && part.file_data) {
        const dataUrl = toImageDataUrl(part.file_data);
        if (dataUrl) images.push(dataUrl);
      }
      return;
    }
    case 'image': {
      const source = part.source;
      if (typeof part.image === 'string' && part.image) {
        images.push(part.image);
      } else if (typeof source === 'string' && source) {
        images.push(source);
      } else if (source && typeof source === 'object') {
        if (source.type === 'base64' && typeof source.data === 'string') {
          const mime = typeof source.media_type === 'string' ? source.media_type : 'image/png';
          images.push(`data:${mime};base64,${source.data}`);
        } else if (typeof source.url === 'string' && source.url) {
          images.push(source.url);
        }
      }
      return;
    }
    default: {
      // Descend into tool results / nested content blocks.
      if (part.content !== undefined) collectImageParts(part.content, images, fromText);
      if (part.output !== undefined) collectImageParts(part.output, images, fromText);
      return;
    }
  }
}

function dedupeSources(sources: ImageSource[]): ImageSource[] {
  const seen = new Set<string>();
  const out: ImageSource[] = [];
  for (const s of sources) {
    const key = typeof s === 'string' ? s : `buf:${s.toString('base64')}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(s);
  }
  return out;
}

/**
 * Extract every inline/attached image from a message list. Works for OpenAI
 * chat-completions (`image_url`), OpenAI Responses (`input_image`,
 * `input_file`) and Anthropic-style (`image.source`) payloads, and also picks
 * up markdown/URL/local-path references embedded in plain text.
 */
/**
 * `fromText: false` takes only structured image parts. An agent's messages carry tool results (READMEs, web
 * pages, file listings) whose markdown images, image URLs and local paths are not meant for the model.
 */
export function extractImagesFromMessages(messages: OpenAIMessage[], fromText: boolean = true): ImageSource[] {
  const images: ImageSource[] = [];
  if (!Array.isArray(messages)) return images;

  for (const msg of messages) {
    if (!msg) continue;
    collectImageParts(msg.content, images, fromText);
  }
  return dedupeSources(images);
}

export async function uploadAndPrepareImage(userToken: string, imageSource: string | Buffer): Promise<string> {
  let fileBuf: Buffer | null = null;
  let baseName = 'image.webp';

  if (Buffer.isBuffer(imageSource)) {
    const tmpInput = `/tmp/ds_in_${Date.now()}_${Math.random().toString(36).slice(2)}.tmp`;
    const tmpWebp = `/tmp/ds_out_${Date.now()}_${Math.random().toString(36).slice(2)}.webp`;
    fs.writeFileSync(tmpInput, imageSource);
    try {
      execSync(`cwebp -q 85 "${tmpInput}" -o "${tmpWebp}" 2>/dev/null || ffmpeg -y -i "${tmpInput}" "${tmpWebp}" 2>/dev/null`);
      fileBuf = fs.readFileSync(tmpWebp);
    } finally {
      if (fs.existsSync(tmpInput)) fs.unlinkSync(tmpInput);
      if (fs.existsSync(tmpWebp)) fs.unlinkSync(tmpWebp);
    }
  } else if (typeof imageSource === 'string') {
    if (imageSource.startsWith('data:image')) {
      const base64Data = imageSource.split(',')[1];
      return uploadAndPrepareImage(userToken, Buffer.from(base64Data, 'base64'));
    } else if (imageSource.startsWith('http://') || imageSource.startsWith('https://')) {
      const downloaded = await downloadImage(imageSource);
      return uploadAndPrepareImage(userToken, downloaded);
    } else if (fs.existsSync(imageSource)) {
      const tmpWebp = `/tmp/ds_out_${Date.now()}_${Math.random().toString(36).slice(2)}.webp`;
      try {
        execSync(`cwebp -q 85 "${imageSource}" -o "${tmpWebp}" 2>/dev/null || ffmpeg -y -i "${imageSource}" "${tmpWebp}" 2>/dev/null`);
        fileBuf = fs.readFileSync(tmpWebp);
      } finally {
        if (fs.existsSync(tmpWebp)) fs.unlinkSync(tmpWebp);
      }
      baseName = path.basename(imageSource, path.extname(imageSource)) + '.webp';
    } else {
      throw new Error(`Image path not found: ${imageSource}`);
    }
  }

  if (!fileBuf) throw new Error('Failed to process image buffer for upload');

  const powHeader = await getPowResponseHeader(userToken, '/api/v0/file/upload_file');
  const boundary = '----WebKitFormBoundary' + Math.random().toString(36).slice(2);
  const prefix = Buffer.from(`--${boundary}\r\nContent-Disposition: form-data; name="file"; filename="${baseName}"\r\nContent-Type: image/webp\r\n\r\n`);
  const suffix = Buffer.from(`\r\n--${boundary}--\r\n`);
  const multipartBody = Buffer.concat([prefix, fileBuf, suffix]);

  const uploadedFileId = await new Promise<string>((resolve, reject) => {
    const req = https.request(
      {
        hostname: config.targetHost,
        port: 443,
        path: '/api/v0/file/upload_file',
        method: 'POST',
        headers: {
          Authorization: `Bearer ${userToken}`,
          'Content-Type': `multipart/form-data; boundary=${boundary}`,
          'Content-Length': multipartBody.length,
          'x-ds-pow-response': powHeader,
          ...config.clientHeaders,
        },
      },
      (res) => {
        let b = '';
        res.on('data', (c) => (b += c));
        res.on('end', () => {
          try {
            const json = JSON.parse(b);
            const fid = json.data?.biz_data?.id;
            if (fid) resolve(fid);
            else reject(new Error(`Upload failed: ${b}`));
          } catch (e) {
            reject(new Error(`Upload response parse error: ${b}`));
          }
        });
      }
    );
    req.on('error', reject);
    req.write(multipartBody);
    req.end();
  });

  const forkedFileId = await new Promise<string>((resolve) => {
    const req = https.request(
      {
        hostname: config.targetHost,
        port: 443,
        path: '/api/v0/file/fork_file_task',
        method: 'POST',
        headers: {
          Authorization: `Bearer ${userToken}`,
          'Content-Type': 'application/json',
          ...config.clientHeaders,
        },
      },
      (res) => {
        let b = '';
        res.on('data', (c) => (b += c));
        res.on('end', () => {
          try {
            const json = JSON.parse(b);
            const fid = json.data?.biz_data?.id;
            if (fid) resolve(fid);
            else resolve(uploadedFileId);
          } catch (e) {
            resolve(uploadedFileId);
          }
        });
      }
    );
    req.on('error', () => resolve(uploadedFileId));
    req.write(JSON.stringify({ file_id: uploadedFileId, to_model_type: 'default' }));
    req.end();
  });

  for (let i = 0; i < 5; i++) {
    const isReady = await new Promise<boolean>((resolve) => {
      const req = https.request(
        {
          hostname: config.targetHost,
          port: 443,
          path: `/api/v0/file/fetch_files?file_ids=${forkedFileId}`,
          method: 'GET',
          headers: {
            Authorization: `Bearer ${userToken}`,
            ...config.clientHeaders,
          },
        },
        (res) => {
          let b = '';
          res.on('data', (c) => (b += c));
          res.on('end', () => {
            try {
              const json = JSON.parse(b);
              const f = json.data?.biz_data?.files?.[0];
              resolve(f?.status === 'SUCCESS' || f?.previewable);
            } catch (e) {
              resolve(true);
            }
          });
        }
      );
      req.on('error', () => resolve(true));
      req.end();
    });
    if (isReady) break;
    await new Promise((r) => setTimeout(r, 800));
  }

  return forkedFileId;
}
