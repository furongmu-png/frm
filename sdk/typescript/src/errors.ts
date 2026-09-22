/**
 * Exception hierarchy for the ZeroDataModel SDK.
 *
 * Every SDK error derives from {@link ZeroDataModelError}, so callers can
 * catch all failures with a single `catch (e) { if (e instanceof
 * ZeroDataModelError) ... }` while still branching on the specific class.
 */

export class ZeroDataModelError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ZeroDataModelError';
  }
}

export class AuthenticationError extends ZeroDataModelError {
  constructor(message: string) {
    super(message);
    this.name = 'AuthenticationError';
  }
}

export class RateLimitError extends ZeroDataModelError {
  /** Suggested wait time in seconds before retrying. */
  retryAfter: number;

  constructor(message: string, retryAfter = 60) {
    super(message);
    this.name = 'RateLimitError';
    this.retryAfter = retryAfter;
  }
}

export class NotFoundError extends ZeroDataModelError {
  constructor(message: string) {
    super(message);
    this.name = 'NotFoundError';
  }
}

export class ValidationError extends ZeroDataModelError {
  constructor(message: string) {
    super(message);
    this.name = 'ValidationError';
  }
}

export class ServerError extends ZeroDataModelError {
  constructor(message: string) {
    super(message);
    this.name = 'ServerError';
  }
}
