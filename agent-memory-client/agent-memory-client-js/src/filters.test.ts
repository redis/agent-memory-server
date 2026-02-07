import { describe, it, expect } from "vitest";
import {
  SessionId,
  Namespace,
  UserId,
  Topics,
  Entities,
  CreatedAt,
  LastAccessed,
  EventDate,
  MemoryType,
} from "./filters";

describe("SessionId", () => {
  it("should create with eq option", () => {
    const filter = new SessionId({ eq: "session-1" });
    expect(filter.toJSON()).toEqual({ eq: "session-1" });
  });

  it("should create with in_ option", () => {
    const filter = new SessionId({ in_: ["s1", "s2"] });
    expect(filter.toJSON()).toEqual({ in_: ["s1", "s2"] });
  });

  it("should create with not_eq option", () => {
    const filter = new SessionId({ not_eq: "session-1" });
    expect(filter.toJSON()).toEqual({ not_eq: "session-1" });
  });

  it("should create with not_in option", () => {
    const filter = new SessionId({ not_in: ["s1", "s2"] });
    expect(filter.toJSON()).toEqual({ not_in: ["s1", "s2"] });
  });

  it("should create with ne option", () => {
    const filter = new SessionId({ ne: "session-1" });
    expect(filter.toJSON()).toEqual({ ne: "session-1" });
  });

  it("should create with startswith option", () => {
    const filter = new SessionId({ startswith: "session-prefix" });
    expect(filter.toJSON()).toEqual({ startswith: "session-prefix" });
  });

  it("should create empty filter", () => {
    const filter = new SessionId();
    expect(filter.toJSON()).toEqual({});
  });
});

describe("Namespace", () => {
  it("should create with eq option", () => {
    const filter = new Namespace({ eq: "ns1" });
    expect(filter.toJSON()).toEqual({ eq: "ns1" });
  });

  it("should create with multiple options", () => {
    const filter = new Namespace({ in_: ["ns1", "ns2"], not_eq: "ns3" });
    expect(filter.toJSON()).toEqual({ in_: ["ns1", "ns2"], not_eq: "ns3" });
  });

  it("should create with not_in option", () => {
    const filter = new Namespace({ not_in: ["ns1", "ns2"] });
    expect(filter.toJSON()).toEqual({ not_in: ["ns1", "ns2"] });
  });

  it("should create with startswith option for hierarchical namespaces", () => {
    const filter = new Namespace({ startswith: "workspace:abc123" });
    expect(filter.toJSON()).toEqual({ startswith: "workspace:abc123" });
  });

  it("should create empty filter", () => {
    const filter = new Namespace();
    expect(filter.toJSON()).toEqual({});
  });
});

describe("UserId", () => {
  it("should create with eq option", () => {
    const filter = new UserId({ eq: "user-1" });
    expect(filter.toJSON()).toEqual({ eq: "user-1" });
  });

  it("should create with in_ option", () => {
    const filter = new UserId({ in_: ["u1", "u2"] });
    expect(filter.toJSON()).toEqual({ in_: ["u1", "u2"] });
  });

  it("should create with not_eq option", () => {
    const filter = new UserId({ not_eq: "user-1" });
    expect(filter.toJSON()).toEqual({ not_eq: "user-1" });
  });

  it("should create with not_in option", () => {
    const filter = new UserId({ not_in: ["u1", "u2"] });
    expect(filter.toJSON()).toEqual({ not_in: ["u1", "u2"] });
  });

  it("should create with startswith option", () => {
    const filter = new UserId({ startswith: "user-prefix" });
    expect(filter.toJSON()).toEqual({ startswith: "user-prefix" });
  });

  it("should create empty filter", () => {
    const filter = new UserId();
    expect(filter.toJSON()).toEqual({});
  });
});

describe("Topics", () => {
  it("should create with any option", () => {
    const filter = new Topics({ any: ["topic1", "topic2"] });
    expect(filter.toJSON()).toEqual({ any: ["topic1", "topic2"] });
  });

  it("should create with all option", () => {
    const filter = new Topics({ all: ["topic1", "topic2"] });
    expect(filter.toJSON()).toEqual({ all: ["topic1", "topic2"] });
  });

  it("should create with none option", () => {
    const filter = new Topics({ none: ["topic1"] });
    expect(filter.toJSON()).toEqual({ none: ["topic1"] });
  });

  it("should create empty filter", () => {
    const filter = new Topics();
    expect(filter.toJSON()).toEqual({});
  });
});

describe("Entities", () => {
  it("should create with any option", () => {
    const filter = new Entities({ any: ["entity1"] });
    expect(filter.toJSON()).toEqual({ any: ["entity1"] });
  });

  it("should create with all option", () => {
    const filter = new Entities({ all: ["entity1", "entity2"] });
    expect(filter.toJSON()).toEqual({ all: ["entity1", "entity2"] });
  });

  it("should create with none option", () => {
    const filter = new Entities({ none: ["excluded"] });
    expect(filter.toJSON()).toEqual({ none: ["excluded"] });
  });

  it("should create empty filter", () => {
    const filter = new Entities();
    expect(filter.toJSON()).toEqual({});
  });
});

describe("CreatedAt", () => {
  it("should create with Date gte option", () => {
    const date = new Date("2024-01-01T00:00:00Z");
    const filter = new CreatedAt({ gte: date });
    expect(filter.toJSON()).toEqual({ gte: date.toISOString() });
  });

  it("should create with string gte option", () => {
    const filter = new CreatedAt({ gte: "2024-01-01T00:00:00Z" });
    expect(filter.toJSON()).toEqual({ gte: "2024-01-01T00:00:00Z" });
  });

  it("should create with Date lte and eq options", () => {
    const date = new Date("2024-12-31T23:59:59Z");
    const filter = new CreatedAt({ lte: date, eq: date });
    expect(filter.toJSON()).toEqual({
      lte: date.toISOString(),
      eq: date.toISOString(),
    });
  });

  it("should create with string lte and eq options", () => {
    const filter = new CreatedAt({ lte: "2024-12-31", eq: "2024-06-15" });
    expect(filter.toJSON()).toEqual({ lte: "2024-12-31", eq: "2024-06-15" });
  });

  it("should create empty filter", () => {
    const filter = new CreatedAt();
    expect(filter.toJSON()).toEqual({});
  });
});

describe("LastAccessed", () => {
  it("should create with Date options", () => {
    const date = new Date("2024-06-15T12:00:00Z");
    const filter = new LastAccessed({ gte: date, lte: date });
    expect(filter.toJSON()).toEqual({
      gte: date.toISOString(),
      lte: date.toISOString(),
    });
  });

  it("should create with string options", () => {
    const filter = new LastAccessed({ gte: "2024-01-01", lte: "2024-12-31", eq: "2024-06-15" });
    expect(filter.toJSON()).toEqual({ gte: "2024-01-01", lte: "2024-12-31", eq: "2024-06-15" });
  });

  it("should create with Date eq option", () => {
    const date = new Date("2024-06-15T12:00:00Z");
    const filter = new LastAccessed({ eq: date });
    expect(filter.toJSON()).toEqual({ eq: date.toISOString() });
  });

  it("should create empty filter", () => {
    const filter = new LastAccessed();
    expect(filter.toJSON()).toEqual({});
  });
});

describe("EventDate", () => {
  it("should create with string options", () => {
    const filter = new EventDate({ gte: "2024-01-01", lte: "2024-12-31" });
    expect(filter.toJSON()).toEqual({ gte: "2024-01-01", lte: "2024-12-31" });
  });

  it("should create with Date options", () => {
    const date = new Date("2024-06-15T12:00:00Z");
    const filter = new EventDate({ gte: date, lte: date, eq: date });
    expect(filter.toJSON()).toEqual({
      gte: date.toISOString(),
      lte: date.toISOString(),
      eq: date.toISOString(),
    });
  });

  it("should create with string eq option", () => {
    const filter = new EventDate({ eq: "2024-06-15" });
    expect(filter.toJSON()).toEqual({ eq: "2024-06-15" });
  });

  it("should create empty filter", () => {
    const filter = new EventDate();
    expect(filter.toJSON()).toEqual({});
  });
});

describe("MemoryType", () => {
  it("should create with eq option", () => {
    const filter = new MemoryType({ eq: "episodic" });
    expect(filter.toJSON()).toEqual({ eq: "episodic" });
  });

  it("should create with in_ option", () => {
    const filter = new MemoryType({ in_: ["episodic", "semantic"] });
    expect(filter.toJSON()).toEqual({ in_: ["episodic", "semantic"] });
  });

  it("should create with not_eq and not_in options", () => {
    const filter = new MemoryType({ not_eq: "message", not_in: ["message"] });
    expect(filter.toJSON()).toEqual({ not_eq: "message", not_in: ["message"] });
  });

  it("should create empty filter", () => {
    const filter = new MemoryType();
    expect(filter.toJSON()).toEqual({});
  });
});
