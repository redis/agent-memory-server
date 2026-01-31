/**
 * Filter classes for search operations.
 *
 * These filters allow for filtering memory search results.
 */

/**
 * Filter by session ID
 */
export class SessionId {
  eq?: string;
  in_?: string[];
  not_eq?: string;
  not_in?: string[];
  ne?: string;

  constructor(options: {
    eq?: string;
    in_?: string[];
    not_eq?: string;
    not_in?: string[];
    ne?: string;
  } = {}) {
    this.eq = options.eq;
    this.in_ = options.in_;
    this.not_eq = options.not_eq;
    this.not_in = options.not_in;
    this.ne = options.ne;
  }

  toJSON(): Record<string, unknown> {
    const result: Record<string, unknown> = {};
    if (this.eq !== undefined) result.eq = this.eq;
    if (this.in_ !== undefined) result.in_ = this.in_;
    if (this.not_eq !== undefined) result.not_eq = this.not_eq;
    if (this.not_in !== undefined) result.not_in = this.not_in;
    if (this.ne !== undefined) result.ne = this.ne;
    return result;
  }
}

/**
 * Filter by namespace
 */
export class Namespace {
  eq?: string;
  in_?: string[];
  not_eq?: string;
  not_in?: string[];

  constructor(options: {
    eq?: string;
    in_?: string[];
    not_eq?: string;
    not_in?: string[];
  } = {}) {
    this.eq = options.eq;
    this.in_ = options.in_;
    this.not_eq = options.not_eq;
    this.not_in = options.not_in;
  }

  toJSON(): Record<string, unknown> {
    const result: Record<string, unknown> = {};
    if (this.eq !== undefined) result.eq = this.eq;
    if (this.in_ !== undefined) result.in_ = this.in_;
    if (this.not_eq !== undefined) result.not_eq = this.not_eq;
    if (this.not_in !== undefined) result.not_in = this.not_in;
    return result;
  }
}

/**
 * Filter by user ID
 */
export class UserId {
  eq?: string;
  in_?: string[];
  not_eq?: string;
  not_in?: string[];

  constructor(options: {
    eq?: string;
    in_?: string[];
    not_eq?: string;
    not_in?: string[];
  } = {}) {
    this.eq = options.eq;
    this.in_ = options.in_;
    this.not_eq = options.not_eq;
    this.not_in = options.not_in;
  }

  toJSON(): Record<string, unknown> {
    const result: Record<string, unknown> = {};
    if (this.eq !== undefined) result.eq = this.eq;
    if (this.in_ !== undefined) result.in_ = this.in_;
    if (this.not_eq !== undefined) result.not_eq = this.not_eq;
    if (this.not_in !== undefined) result.not_in = this.not_in;
    return result;
  }
}

/**
 * Filter by topics
 */
export class Topics {
  any?: string[];
  all?: string[];
  none?: string[];

  constructor(options: {
    any?: string[];
    all?: string[];
    none?: string[];
  } = {}) {
    this.any = options.any;
    this.all = options.all;
    this.none = options.none;
  }

  toJSON(): Record<string, unknown> {
    const result: Record<string, unknown> = {};
    if (this.any !== undefined) result.any = this.any;
    if (this.all !== undefined) result.all = this.all;
    if (this.none !== undefined) result.none = this.none;
    return result;
  }
}

/**
 * Filter by entities
 */
export class Entities {
  any?: string[];
  all?: string[];
  none?: string[];

  constructor(options: {
    any?: string[];
    all?: string[];
    none?: string[];
  } = {}) {
    this.any = options.any;
    this.all = options.all;
    this.none = options.none;
  }

  toJSON(): Record<string, unknown> {
    const result: Record<string, unknown> = {};
    if (this.any !== undefined) result.any = this.any;
    if (this.all !== undefined) result.all = this.all;
    if (this.none !== undefined) result.none = this.none;
    return result;
  }
}

/**
 * Filter by creation date
 */
export class CreatedAt {
  gte?: Date | string;
  lte?: Date | string;
  eq?: Date | string;

  constructor(options: {
    gte?: Date | string;
    lte?: Date | string;
    eq?: Date | string;
  } = {}) {
    this.gte = options.gte;
    this.lte = options.lte;
    this.eq = options.eq;
  }

  toJSON(): Record<string, unknown> {
    const result: Record<string, unknown> = {};
    if (this.gte !== undefined)
      result.gte = this.gte instanceof Date ? this.gte.toISOString() : this.gte;
    if (this.lte !== undefined)
      result.lte = this.lte instanceof Date ? this.lte.toISOString() : this.lte;
    if (this.eq !== undefined)
      result.eq = this.eq instanceof Date ? this.eq.toISOString() : this.eq;
    return result;
  }
}

/**
 * Filter by last accessed date
 */
export class LastAccessed {
  gte?: Date | string;
  lte?: Date | string;
  eq?: Date | string;

  constructor(options: {
    gte?: Date | string;
    lte?: Date | string;
    eq?: Date | string;
  } = {}) {
    this.gte = options.gte;
    this.lte = options.lte;
    this.eq = options.eq;
  }

  toJSON(): Record<string, unknown> {
    const result: Record<string, unknown> = {};
    if (this.gte !== undefined)
      result.gte = this.gte instanceof Date ? this.gte.toISOString() : this.gte;
    if (this.lte !== undefined)
      result.lte = this.lte instanceof Date ? this.lte.toISOString() : this.lte;
    if (this.eq !== undefined)
      result.eq = this.eq instanceof Date ? this.eq.toISOString() : this.eq;
    return result;
  }
}

/**
 * Filter by event date
 */
export class EventDate {
  gte?: Date | string;
  lte?: Date | string;
  eq?: Date | string;

  constructor(options: {
    gte?: Date | string;
    lte?: Date | string;
    eq?: Date | string;
  } = {}) {
    this.gte = options.gte;
    this.lte = options.lte;
    this.eq = options.eq;
  }

  toJSON(): Record<string, unknown> {
    const result: Record<string, unknown> = {};
    if (this.gte !== undefined)
      result.gte = this.gte instanceof Date ? this.gte.toISOString() : this.gte;
    if (this.lte !== undefined)
      result.lte = this.lte instanceof Date ? this.lte.toISOString() : this.lte;
    if (this.eq !== undefined)
      result.eq = this.eq instanceof Date ? this.eq.toISOString() : this.eq;
    return result;
  }
}

/**
 * Filter by memory type
 */
export class MemoryType {
  eq?: string;
  in_?: string[];
  not_eq?: string;
  not_in?: string[];

  constructor(options: {
    eq?: string;
    in_?: string[];
    not_eq?: string;
    not_in?: string[];
  } = {}) {
    this.eq = options.eq;
    this.in_ = options.in_;
    this.not_eq = options.not_eq;
    this.not_in = options.not_in;
  }

  toJSON(): Record<string, unknown> {
    const result: Record<string, unknown> = {};
    if (this.eq !== undefined) result.eq = this.eq;
    if (this.in_ !== undefined) result.in_ = this.in_;
    if (this.not_eq !== undefined) result.not_eq = this.not_eq;
    if (this.not_in !== undefined) result.not_in = this.not_in;
    return result;
  }
}
