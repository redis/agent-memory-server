import { describe, it, expect } from "vitest";
import { generateId, MemoryTypeEnum } from "./models";

describe("generateId", () => {
  it("should generate a ULID string", () => {
    const id = generateId();
    expect(typeof id).toBe("string");
    expect(id.length).toBe(26);
  });

  it("should generate unique IDs", () => {
    const id1 = generateId();
    const id2 = generateId();
    expect(id1).not.toBe(id2);
  });
});

describe("MemoryTypeEnum", () => {
  it("should have expected values", () => {
    expect(MemoryTypeEnum.EPISODIC).toBe("episodic");
    expect(MemoryTypeEnum.SEMANTIC).toBe("semantic");
    expect(MemoryTypeEnum.MESSAGE).toBe("message");
  });
});
