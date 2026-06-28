use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::Command;

use aquaregia::{Tool, tool};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

const MAX_TOOL_OUTPUT_CHARS: usize = 12_000;
const MAX_READ_LIMIT: u64 = 1_000;

#[derive(Debug, Deserialize, JsonSchema)]
struct BashArgs {
    command: String,
}

pub fn bash() -> Tool {
    tool("bash")
        .description("Execute a shell command in the current workspace")
        .try_execute(|args: BashArgs| async move {
            if is_dangerous_command(&args.command) {
                return Err(aquaregia::tool::ToolExecError::Execution(format!(
                    "blocked dangerous command: {}",
                    args.command
                )));
            }

            let output = Command::new("sh")
                .arg("-lc")
                .arg(&args.command)
                .output()
                .map_err(|e| {
                    aquaregia::tool::ToolExecError::Execution(format!("bash execution failed: {e}"))
                })?;

            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            let merged = if stderr.is_empty() {
                stdout
            } else {
                format!("{stdout}\n[stderr]\n{stderr}")
            };

            Ok(json!({
                "command": args.command,
                "exit_code": output.status.code().unwrap_or(-1),
                "output": truncate_text(merged.trim(), MAX_TOOL_OUTPUT_CHARS)
            }))
        })
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ReadArgs {
    path: String,
    offset: Option<u64>,
    limit: Option<u64>,
}

pub fn read() -> Tool {
    tool("read")
        .description("Read a file with an optional line window")
        .try_execute(|args: ReadArgs| async move {
            let offset = args.offset.unwrap_or(0) as usize;
            let limit = args.limit.unwrap_or(200);
            if limit == 0 || limit > MAX_READ_LIMIT {
                return Err(aquaregia::tool::ToolExecError::Execution(format!(
                    "`limit` must be in [1, {MAX_READ_LIMIT}]"
                )));
            }

            let safe_path =
                resolve_safe_path(&args.path).map_err(aquaregia::tool::ToolExecError::Execution)?;
            let text = fs::read_to_string(&safe_path).map_err(|e| {
                aquaregia::tool::ToolExecError::Execution(format!(
                    "read failed for `{}`: {e}",
                    args.path
                ))
            })?;

            let lines = text.lines().collect::<Vec<_>>();
            let start = offset.min(lines.len());
            let end = start.saturating_add(limit as usize).min(lines.len());
            let body = lines[start..end]
                .iter()
                .enumerate()
                .map(|(i, line)| format!("{}\t{}", start + i + 1, line))
                .collect::<Vec<_>>()
                .join("\n");

            Ok(json!({
                "path": args.path,
                "line_start": start + 1,
                "line_end": end,
                "total_lines": lines.len(),
                "content": truncate_text(&body, MAX_TOOL_OUTPUT_CHARS)
            }))
        })
}

#[derive(Debug, Deserialize, JsonSchema)]
struct WriteArgs {
    path: String,
    content: String,
}

pub fn write() -> Tool {
    tool("write")
        .description("Write full file content, creating parent directories if needed")
        .try_execute(|args: WriteArgs| async move {
            let safe_path =
                resolve_safe_path(&args.path).map_err(aquaregia::tool::ToolExecError::Execution)?;

            if let Some(parent) = safe_path.parent() {
                fs::create_dir_all(parent).map_err(|e| {
                    aquaregia::tool::ToolExecError::Execution(format!(
                        "create parent dirs failed for `{}`: {e}",
                        args.path
                    ))
                })?;
            }

            fs::write(&safe_path, args.content.as_bytes()).map_err(|e| {
                aquaregia::tool::ToolExecError::Execution(format!(
                    "write failed for `{}`: {e}",
                    args.path
                ))
            })?;

            Ok(json!({
                "path": args.path,
                "bytes_written": args.content.len()
            }))
        })
}

#[derive(Debug, Deserialize, JsonSchema)]
struct EditArgs {
    path: String,
    old_string: String,
    new_string: String,
}

pub fn edit() -> Tool {
    tool("edit")
        .description("Replace one unique old_string with new_string in a file")
        .try_execute(|args: EditArgs| async move {
            let safe_path =
                resolve_safe_path(&args.path).map_err(aquaregia::tool::ToolExecError::Execution)?;

            let original = fs::read_to_string(&safe_path).map_err(|e| {
                aquaregia::tool::ToolExecError::Execution(format!(
                    "read failed for `{}`: {e}",
                    args.path
                ))
            })?;
            let occurrences = original.matches(&args.old_string).count();

            if occurrences == 0 {
                return Err(aquaregia::tool::ToolExecError::Execution(format!(
                    "old_string not found in `{}`",
                    args.path
                )));
            }
            if occurrences > 1 {
                return Err(aquaregia::tool::ToolExecError::Execution(format!(
                    "old_string appears {occurrences} times in `{}`",
                    args.path
                )));
            }

            let updated = original.replacen(&args.old_string, &args.new_string, 1);
            fs::write(&safe_path, updated.as_bytes()).map_err(|e| {
                aquaregia::tool::ToolExecError::Execution(format!(
                    "write failed for `{}`: {e}",
                    args.path
                ))
            })?;

            Ok(json!({
                "path": args.path,
                "replaced": true
            }))
        })
}

fn resolve_safe_path(input_path: &str) -> Result<PathBuf, String> {
    let cwd = std::env::current_dir().map_err(|e| format!("cannot get cwd: {e}"))?;
    let joined = if Path::new(input_path).is_absolute() {
        PathBuf::from(input_path)
    } else {
        cwd.join(input_path)
    };
    let normalized = normalize_path(&joined);

    if !normalized.starts_with(&cwd) {
        return Err(format!("path escapes workspace: `{input_path}`"));
    }

    Ok(normalized)
}

fn normalize_path(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            Component::Prefix(prefix) => out.push(prefix.as_os_str()),
            Component::RootDir => out.push(component.as_os_str()),
            Component::Normal(part) => out.push(part),
        }
    }
    out
}

fn is_dangerous_command(command: &str) -> bool {
    let lowered = command.to_ascii_lowercase();
    let blocked = [
        "rm -rf /",
        "rm -rf ~",
        "shutdown",
        "reboot",
        "halt",
        "mkfs.",
        "dd if=",
        "git reset --hard",
    ];
    blocked.iter().any(|pattern| lowered.contains(pattern))
}

fn truncate_text(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }

    let mut truncated = text.chars().take(max_chars).collect::<String>();
    truncated.push_str("\n...[truncated]...");
    truncated
}
