import { describe, it, expect } from "vitest";
import { generateId, MemoryTypeEnum, type MemoryRecord } from "./models";

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

describe("MemoryRecord", () => {
  it("should allow extraction metadata fields", () => {
    const record: MemoryRecord = {
      id: "mem-1",
      text: "Thread summary",
      extraction_strategy: "summary",
      extraction_strategy_config: { summary_version: "v1" },
      metadata: { message_count: 2 },
    };

    expect(record.extraction_strategy).toBe("summary");
    expect(record.extraction_strategy_config).toEqual({ summary_version: "v1" });
    expect(record.metadata).toEqual({ message_count: 2 });
  });
});
