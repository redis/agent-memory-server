/**
 * Exception classes for the Agent Memory Client.
 */

/**
 * Base error for all memory client errors.
 */
export class MemoryClientError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "MemoryClientError";
    // Maintains proper stack trace for where our error was thrown (only in V8)
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, MemoryClientError);
    }
  }
}

/**
 * Raised when memory record or filter validation fails.
 */
export class MemoryValidationError extends MemoryClientError {
  constructor(message: string) {
    super(message);
    this.name = "MemoryValidationError";
  }
}

/**
 * Raised when a requested memory or session is not found.
 */
export class MemoryNotFoundError extends MemoryClientError {
  constructor(message: string) {
    super(message);
    this.name = "MemoryNotFoundError";
  }
}

/**
 * Raised when the memory server returns an error.
 */
export class MemoryServerError extends MemoryClientError {
  statusCode?: number;

  constructor(message: string, statusCode?: number) {
    super(message);
    this.name = "MemoryServerError";
    this.statusCode = statusCode;
  }
}
