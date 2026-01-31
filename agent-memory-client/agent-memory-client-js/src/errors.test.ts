import { describe, it, expect } from "vitest";
import {
  MemoryClientError,
  MemoryValidationError,
  MemoryNotFoundError,
  MemoryServerError,
} from "./errors";

describe("MemoryClientError", () => {
  it("should create error with message", () => {
    const error = new MemoryClientError("test error");
    expect(error.message).toBe("test error");
    expect(error.name).toBe("MemoryClientError");
    expect(error).toBeInstanceOf(Error);
    expect(error).toBeInstanceOf(MemoryClientError);
  });

  it("should have proper stack trace", () => {
    const error = new MemoryClientError("test error");
    expect(error.stack).toBeDefined();
  });

  it("should work when Error.captureStackTrace is undefined", () => {
    // Save the original
    const originalCaptureStackTrace = Error.captureStackTrace;

    // Remove captureStackTrace to simulate non-V8 environment
    // @ts-expect-error - intentionally removing to test branch
    delete Error.captureStackTrace;

    try {
      const error = new MemoryClientError("test error");
      expect(error.message).toBe("test error");
      expect(error.name).toBe("MemoryClientError");
      expect(error).toBeInstanceOf(Error);
    } finally {
      // Restore the original
      Error.captureStackTrace = originalCaptureStackTrace;
    }
  });
});

describe("MemoryValidationError", () => {
  it("should create error with message", () => {
    const error = new MemoryValidationError("validation failed");
    expect(error.message).toBe("validation failed");
    expect(error.name).toBe("MemoryValidationError");
    expect(error).toBeInstanceOf(MemoryClientError);
  });
});

describe("MemoryNotFoundError", () => {
  it("should create error with message", () => {
    const error = new MemoryNotFoundError("not found");
    expect(error.message).toBe("not found");
    expect(error.name).toBe("MemoryNotFoundError");
    expect(error).toBeInstanceOf(MemoryClientError);
  });
});

describe("MemoryServerError", () => {
  it("should create error with message and status code", () => {
    const error = new MemoryServerError("server error", 500);
    expect(error.message).toBe("server error");
    expect(error.name).toBe("MemoryServerError");
    expect(error.statusCode).toBe(500);
    expect(error).toBeInstanceOf(MemoryClientError);
  });

  it("should create error without status code", () => {
    const error = new MemoryServerError("server error");
    expect(error.message).toBe("server error");
    expect(error.statusCode).toBeUndefined();
  });
});
