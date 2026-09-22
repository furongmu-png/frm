/**
 * Configuration for the ZeroDataModel SDK client.
 */

export interface ClientConfigOptions {
  /** Base URL of the ZeroDataModel REST API (trailing slash is stripped). */
  baseUrl?: string;
  /** Optional API key sent as the `X-API-Key` header. */
  apiKey?: string;
  /** Per-request timeout in milliseconds. */
  timeout?: number;
  /** Maximum number of retries for retriable failures. */
  maxRetries?: number;
  /** Base backoff (ms) between retries. */
  retryBackoff?: number;
  /**
   * Custom `fetch` implementation. Defaults to the global `fetch`. Exposed
   * primarily so the test-suite can inject a mock without monkey-patching
   * globals.
   */
  fetch?: typeof fetch;
}

export class ClientConfig {
  baseUrl: string;
  apiKey?: string;
  timeout: number;
  maxRetries: number;
  retryBackoff: number;
  fetch?: typeof fetch;

  constructor(config: ClientConfigOptions = {}) {
    this.baseUrl = (config.baseUrl ?? 'http://localhost:8000').replace(/\/$/, '');
    this.apiKey = config.apiKey;
    this.timeout = config.timeout ?? 30000;
    this.maxRetries = config.maxRetries ?? 3;
    this.retryBackoff = config.retryBackoff ?? 500;
    this.fetch = config.fetch;
  }
}
