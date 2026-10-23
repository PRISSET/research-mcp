import http from 'http';
import { config } from './config';
import { handleRoutes } from './routes/router';

const server = http.createServer(async (req, res) => {
  await handleRoutes(req, res);
});

server.listen(config.port, '127.0.0.1', () => {
  console.log(`[proxy] DeepSeek Web Chat Proxy listening on 127.0.0.1:${config.port}`);
});

process.on('SIGTERM', () => server.close(() => process.exit(0)));
process.on('SIGINT', () => server.close(() => process.exit(0)));
