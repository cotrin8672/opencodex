import type { AdapterEvent } from "../../types";

type ErrorDetails = Pick<Extract<AdapterEvent, { type: "error" }>, "usage" | "errorType">;

export class ClaudeSessionError extends Error {
  constructor(
    message: string,
    readonly code: string,
    readonly status = 409,
    readonly retryable = false,
    readonly details: ErrorDetails = {},
  ) { super(message); }
}
