import WebSocket from 'ws';
import { InboundFromPython, OutboundToPython, inboundSchema } from './messages.js';
import { debug } from './debug.js';

export class PythonWsClient {
	private ws?: WebSocket;
	private url: string;
	private reconnectDelay = 3000;
	private shouldRun = true;
	private onOpenCb?: () => void;

	constructor(
		url: string,
		private onMessage: (msg: InboundFromPython) => void,
	) {
		this.url = url;
		debug('PythonWsClient created for URL:', url);
	}

	setOnOpen(cb: () => void) {
		this.onOpenCb = cb;
	}

	start() {
		this.connect();
		debug('PythonWsClient started.');
	}
	stop() {
		this.shouldRun = false;
		this.ws?.close();
		debug('PythonWsClient stopped.');
	}

	send(msg: OutboundToPython) {
		if (this.ws && this.ws.readyState === WebSocket.OPEN) {
			this.ws.send(JSON.stringify(msg));
			debug(`Sent message ${msg.type} to Python service`);
		} else {
			debug('Could not send message, WebSocket not open. State:', this.ws?.readyState);
		}
	}

	private connect() {
		this.ws = new WebSocket(this.url);
		debug('Attempting to connect to WebSocket:', this.url);
		this.ws.on('open', () => {
			console.log('[ws] connected');
			debug('WebSocket connected.');
			if (this.onOpenCb) {
				this.onOpenCb();
			}
		});
		this.ws.on('message', (data) => {
			const parsed = JSON.parse(data.toString());
			debug('Received raw message from WebSocket:', parsed);
			const validated = inboundSchema.safeParse(parsed);
			if (!validated.success) {
				debug('Invalid message received, validation failed:', validated.error);
				throw new Error('Invalid message received');
			}
			this.onMessage(validated.data as InboundFromPython);
		});
		this.ws.on('close', () => {
			console.log('[ws] closed');
			debug('WebSocket closed.');
			if (this.shouldRun) {
				debug(`Will attempt to reconnect in ${this.reconnectDelay}ms.`);
				setTimeout(() => this.connect(), this.reconnectDelay);
			}
		});
		this.ws.on('error', (err) => {
			console.error('[ws] error', err.message);
			debug('WebSocket error:', err);
		});
	}
}
