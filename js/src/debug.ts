
import { format } from 'util';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const debug = (...args: any[]) => {
  if (process.env.DEBUG === '1') {
    const message = format(...args);
    const timestamp = new Date().toISOString();
    console.log(`[${timestamp}] [DEBUG]`, message);
  }
};
