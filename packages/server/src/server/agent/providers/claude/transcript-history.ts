import fs from "node:fs";
import path from "node:path";
import { normalizeProviderReplayTimestamp } from "../../provider-history-timestamps.js";
import {
  parseClaudeSubagentMeta,
  type ClaudeReplayEntry,
  type ClaudeSubagentMeta,
} from "./subagents/replay-source.js";
import {
  parseClaudeWorkflowRun,
  type ClaudeWorkflowRun,
} from "./subagents/workflow-replay-source.js";

interface TranscriptEntry extends ClaudeReplayEntry {
  isSidechain?: unknown;
}

interface RecordLocation {
  file: string;
  offset: number;
  length: number;
  timestamp: string | null;
}

/** Chunk in bytes so UTF-8 and multi-chunk records are decoded once, at a line boundary. */
function* locatedRecords(
  file: string,
): Generator<{ entry: TranscriptEntry; location: RecordLocation }> {
  const fd = fs.openSync(file, "r");
  try {
    const stat = fs.fstatSync(fd);
    // Opening a directory succeeds on some platforms; it must never look like empty history.
    if (stat.isDirectory()) {
      throw Object.assign(new Error(`EISDIR: illegal operation on a directory, read '${file}'`), {
        code: "EISDIR",
        syscall: "read",
        path: file,
      });
    }
    let remaining = stat.size;
    let offset = 0;
    let pieces: Buffer[] = [];
    let length = 0;
    while (remaining > 0) {
      const buffer = Buffer.allocUnsafe(Math.min(64 * 1024, remaining));
      const count = fs.readSync(fd, buffer, 0, buffer.length, null);
      if (count === 0) break;
      remaining -= count;
      let start = 0;
      let end: number;
      while ((end = buffer.indexOf(10, start)) !== -1 && end < count) {
        pieces.push(buffer.subarray(start, end));
        length += end - start;
        const entry = parseRecord(Buffer.concat(pieces, length).toString("utf8"));
        const location = {
          file,
          offset,
          length,
          timestamp: normalizeProviderReplayTimestamp(entry?.timestamp),
        };
        offset += length + 1;
        pieces = [];
        length = 0;
        if (entry) yield { entry, location };
        start = end + 1;
      }
      if (start < count) {
        pieces.push(buffer.subarray(start, count));
        length += count - start;
      }
    }
    const entry = parseRecord(Buffer.concat(pieces, length).toString("utf8"));
    if (entry)
      yield {
        entry,
        location: {
          file,
          offset,
          length,
          timestamp: normalizeProviderReplayTimestamp(entry.timestamp),
        },
      };
  } finally {
    fs.closeSync(fd);
  }
}

/** Repeatable sources let ownership be resolved before replay without retaining raw records. */
function records(file: string): Iterable<TranscriptEntry> {
  return {
    *[Symbol.iterator]() {
      for (const { entry } of locatedRecords(file)) yield entry;
    },
  };
}

/** Workflow children share a timeline. Sort byte locations, then parse in that order so tool
 * results see their declarations even when directory order differs from chronological order. */
function workflowRecords(files: readonly string[]): Iterable<TranscriptEntry> {
  return {
    *[Symbol.iterator]() {
      const locations: RecordLocation[] = [];
      for (const file of files)
        for (const { entry, location } of locatedRecords(file)) {
          if (entry.type !== "user" || isToolResult(entry)) locations.push(location);
        }
      locations.sort((a, b) => {
        if (!a.timestamp && !b.timestamp) return 0;
        if (!a.timestamp) return 1;
        if (!b.timestamp) return -1;
        return Date.parse(a.timestamp) - Date.parse(b.timestamp);
      });
      let currentFile: string | null = null;
      let fd: number | null = null;
      try {
        for (const location of locations) {
          if (fd === null || currentFile !== location.file) {
            if (fd !== null) fs.closeSync(fd);
            fd = null;
            currentFile = location.file;
            fd = fs.openSync(currentFile, "r");
          }
          const buffer = Buffer.allocUnsafe(location.length);
          let read = 0;
          while (read < buffer.length) {
            const count = fs.readSync(
              fd,
              buffer,
              read,
              buffer.length - read,
              location.offset + read,
            );
            if (count === 0) break;
            read += count;
          }
          const entry = parseRecord(buffer.subarray(0, read).toString("utf8"));
          if (entry) yield entry;
        }
      } finally {
        if (fd !== null) fs.closeSync(fd);
      }
    },
  };
}

function isToolResult(entry: TranscriptEntry): boolean {
  return (
    Array.isArray(entry.message?.content) &&
    entry.message.content.some((block) => block?.type === "tool_result")
  );
}

function parseRecord(line: string): TranscriptEntry | null {
  try {
    const value: unknown = JSON.parse(line);
    return value !== null && typeof value === "object" && !Array.isArray(value)
      ? (value as TranscriptEntry)
      : null;
  } catch {
    return null;
  }
}

function selectRecords(
  files: readonly string[],
  accept: (entry: TranscriptEntry) => boolean,
): Iterable<TranscriptEntry> {
  return {
    *[Symbol.iterator]() {
      for (const file of files) for (const entry of records(file)) if (accept(entry)) yield entry;
    },
  };
}

function readOptionalFile(file: string): string | null {
  try {
    return fs.readFileSync(file, "utf8");
  } catch {
    return null;
  }
}

/** Discover sources without retaining transcript contents. Unrelated sidechains stay excluded
 * by the replay ownership resolver; workflow children retain their separate run ownership. */
export function readClaudeReplayHistory(historyPath: string): {
  parentEntries: Iterable<TranscriptEntry>;
  subagents: {
    agentId: string;
    meta: ClaudeSubagentMeta | null;
    entries: Iterable<TranscriptEntry>;
  }[];
  workflows: ClaudeWorkflowRun[];
  workflowEntriesByRunId: Map<string, Iterable<TranscriptEntry>>;
} {
  const sessionDirectory = path.join(
    path.dirname(historyPath),
    path.basename(historyPath, ".jsonl"),
  );
  const sidechainDirectory = path.join(sessionDirectory, "subagents");
  const sidechainFiles = [historyPath];
  const workflowFiles = new Map<string, string[]>();
  const metaByAgentId = new Map<string, ClaudeSubagentMeta>();
  const directories = [sidechainDirectory];
  while (directories.length) {
    const directory = directories.pop()!;
    if (!fs.existsSync(directory)) continue;
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
      const file = path.join(directory, entry.name);
      if (entry.isDirectory()) {
        directories.push(file);
        continue;
      }
      if (!entry.isFile()) continue;
      if (entry.name.endsWith(".jsonl")) {
        const parts = path.relative(sidechainDirectory, file).split(path.sep);
        if (parts[0] === "workflows" && parts.length >= 3) {
          const runId = parts[1]!;
          const files = workflowFiles.get(runId) ?? [];
          files.push(file);
          workflowFiles.set(runId, files);
        } else sidechainFiles.push(file);
        continue;
      }
      const agentId = /^agent-(.+)\.meta\.json$/.exec(entry.name)?.[1];
      if (!agentId) continue;
      const contents = readOptionalFile(file);
      const meta = contents === null ? null : parseClaudeSubagentMeta(contents);
      if (meta) metaByAgentId.set(agentId, meta);
    }
  }
  const filesByAgentId = new Map<string, string[]>();
  for (const file of sidechainFiles) {
    const agentsInFile = new Set<string>();
    for (const entry of records(file)) {
      if (entry.isSidechain === true && typeof entry.agentId === "string")
        agentsInFile.add(entry.agentId);
    }
    for (const agentId of agentsInFile) {
      const files = filesByAgentId.get(agentId) ?? [];
      files.push(file);
      filesByAgentId.set(agentId, files);
    }
  }
  return {
    parentEntries: selectRecords([historyPath], (entry) => entry.isSidechain !== true),
    subagents: [...filesByAgentId].map(([agentId, files]) => ({
      agentId,
      meta: metaByAgentId.get(agentId) ?? null,
      entries: selectRecords(
        files,
        (entry) => entry.isSidechain === true && entry.agentId === agentId,
      ),
    })),
    workflows: readWorkflows(sessionDirectory),
    workflowEntriesByRunId: new Map(
      [...workflowFiles].map(([runId, files]) => [runId, workflowRecords(files)]),
    ),
  };
}

function readWorkflows(sessionDirectory: string): ClaudeWorkflowRun[] {
  const workflows: ClaudeWorkflowRun[] = [];
  const workflowDirectory = path.join(sessionDirectory, "workflows");
  if (fs.existsSync(workflowDirectory)) {
    for (const entry of fs.readdirSync(workflowDirectory, { withFileTypes: true })) {
      if (!entry.isFile() || !entry.name.endsWith(".json")) continue;
      const contents = readOptionalFile(path.join(workflowDirectory, entry.name));
      const workflow = contents === null ? null : parseClaudeWorkflowRun(contents);
      if (workflow) workflows.push(workflow);
    }
  }
  return workflows;
}
