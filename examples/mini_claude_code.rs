use aquaregia::{Agent, AgentStep, LlmClient, Message, tool};
use serde_json::{Value, json};
use std::fs;
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};
use std::process::Command;

const DEFAULT_MODEL: &str = "deepseek-chat";
const DEFAULT_DEEPSEEK_BASE_URL: &str = "https://api.deepseek.com";
const MAX_STEPS: u8 = 12;
const MAX_TOOL_OUTPUT_CHARS: usize = 12_000;
const MAX_READ_LIMIT: u64 = 1_000;
const SYSTEM_PROMPT: &str = r#"
你是 mini-claude-code，一个运行在用户本地终端中的代码助手。
你可以使用 4 个工具：bash、read、write、edit。

工作原则：
1. 先读后改：修改文件前，优先 read 确认上下文。
2. 最小改动：只执行当前任务所需操作，不做无关修改。
3. 局部优先：能 edit 就不 write 全量覆盖。
4. 结果清晰：回答里给出做了什么、改了哪些文件、下一步建议。
5. 使用中文与用户交流。
"#;

/// 场景：最小可运行的终端 Code Agent（TUI + 工具 + 系统提示词）。
///
/// 运行：
/// DEEPSEEK_API_KEY=... cargo run --example mini_claude_code
/// 可选：
/// DEEPSEEK_BASE_URL=https://api.deepseek.com DEEPSEEK_MODEL=deepseek-chat cargo run --example mini_claude_code
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("DEEPSEEK_API_KEY").map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "missing DEEPSEEK_API_KEY; set it before running this example",
        )
    })?;
    let base_url = std::env::var("DEEPSEEK_BASE_URL")
        .unwrap_or_else(|_| DEFAULT_DEEPSEEK_BASE_URL.to_string());
    let model = std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());

    let client = LlmClient::openai_compatible(base_url.clone())
        .api_key(api_key)
        .build()?;
    let agent = Agent::builder(client, model.clone())
        .tools([bash])
        .tools([read])
        .tools([write])
        .tools([edit])
        .max_steps(MAX_STEPS)
        .temperature(0.2)
        .max_output_tokens(1_400)
        .on_step_finish(print_step_debug)
        .build()?;

    println!("mini_claude_code (aquaregia example)");
    println!("model: openai-compatible/{}", model);
    println!("base_url: {}", base_url);
    println!("cwd: {}", std::env::current_dir()?.display());
    println!("exit: Ctrl+C or Ctrl+D");

    let mut history = vec![Message::system_text(SYSTEM_PROMPT)];

    loop {
        print!("\n> ");
        io::stdout().flush()?;

        let mut input = String::new();
        if io::stdin().read_line(&mut input)? == 0 {
            println!("\nEOF, bye.");
            break;
        }

        let question = input.trim();
        if question.is_empty() {
            continue;
        }

        let mut messages = history.clone();
        messages.push(Message::user_text(question.to_string()));

        match agent.run_messages(messages).await {
            Ok(result) => {
                println!("\n{}", result.output_text.trim());
                println!(
                    "\n[steps={} tokens in/out/total = {}/{}/{}]",
                    result.steps,
                    result.usage_total.input_tokens,
                    result.usage_total.output_tokens,
                    result.usage_total.total_tokens
                );
                history = result.transcript;
            }
            Err(err) => {
                eprintln!("\n[error] {}", err);
            }
        }
    }

    Ok(())
}

#[tool(description = "Execute a shell command in current workspace")]
async fn bash(command: String) -> Result<Value, String> {
    if is_dangerous_command(&command) {
        return Err(format!("blocked dangerous command: {}", command));
    }

    let output = Command::new("sh")
        .arg("-lc")
        .arg(&command)
        .output()
        .map_err(|e| format!("bash execution failed: {}", e))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let merged = format!(
        "{}{}",
        stdout,
        if stderr.is_empty() {
            String::new()
        } else {
            format!("\n[stderr]\n{}", stderr)
        }
    );

    Ok(json!({
        "command": command,
        "exit_code": output.status.code().unwrap_or(-1),
        "output": truncate_text(merged.trim(), MAX_TOOL_OUTPUT_CHARS)
    }))
}

#[tool(description = "Read a file with optional line window")]
async fn read(path: String, offset: Option<u64>, limit: Option<u64>) -> Result<Value, String> {
    let offset = offset.unwrap_or(0) as usize;
    let limit = limit.unwrap_or(200);
    if limit == 0 || limit > MAX_READ_LIMIT {
        return Err(format!("`limit` must be in [1, {}]", MAX_READ_LIMIT));
    }
    let safe_path = resolve_safe_path(&path)?;

    let text =
        fs::read_to_string(&safe_path).map_err(|e| format!("read failed for `{}`: {}", path, e))?;
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
        "path": path,
        "line_start": start + 1,
        "line_end": end,
        "total_lines": lines.len(),
        "content": truncate_text(&body, MAX_TOOL_OUTPUT_CHARS)
    }))
}

#[tool(description = "Write full file content (create parent dirs automatically)")]
async fn write(path: String, content: String) -> Result<Value, String> {
    let safe_path = resolve_safe_path(&path)?;

    if let Some(parent) = safe_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("create parent dirs failed for `{}`: {}", path, e))?;
    }

    fs::write(&safe_path, content.as_bytes())
        .map_err(|e| format!("write failed for `{}`: {}", path, e))?;

    Ok(json!({
        "path": path,
        "bytes_written": content.len()
    }))
}

#[tool(description = "Edit file by replacing one unique old_string with new_string")]
async fn edit(path: String, old_string: String, new_string: String) -> Result<Value, String> {
    let safe_path = resolve_safe_path(&path)?;

    let original =
        fs::read_to_string(&safe_path).map_err(|e| format!("read failed for `{}`: {}", path, e))?;
    let occurrences = original.matches(&old_string).count();

    if occurrences == 0 {
        return Err(format!("old_string not found in `{}`", path));
    }
    if occurrences > 1 {
        return Err(format!(
            "old_string appears {} times in `{}`, please provide a unique snippet",
            occurrences, path
        ));
    }

    let updated = original.replacen(&old_string, &new_string, 1);
    fs::write(&safe_path, updated.as_bytes())
        .map_err(|e| format!("write failed for `{}`: {}", path, e))?;

    Ok(json!({
        "path": path,
        "replaced": true
    }))
}

fn resolve_safe_path(input_path: &str) -> Result<PathBuf, String> {
    let cwd = std::env::current_dir().map_err(|e| format!("cannot get cwd: {}", e))?;
    let joined = if Path::new(input_path).is_absolute() {
        PathBuf::from(input_path)
    } else {
        cwd.join(input_path)
    };
    let normalized = normalize_path(&joined);

    if !normalized.starts_with(&cwd) {
        return Err(format!("path escapes workspace: `{}`", input_path));
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
    blocked.iter().any(|p| lowered.contains(p))
}

fn truncate_text(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }

    let mut truncated = String::new();
    for ch in text.chars().take(max_chars) {
        truncated.push(ch);
    }
    truncated.push_str("\n...[truncated]...");
    truncated
}

fn print_step_debug(step: &AgentStep) {
    println!("\n--- step {} ---", step.step);
    println!(
        "finish_reason={:?} usage={}/{}/{}",
        step.finish_reason,
        step.usage.input_tokens,
        step.usage.output_tokens,
        step.usage.total_tokens
    );
    if !step.output_text.trim().is_empty() {
        println!(
            "assistant: {}",
            one_line(&truncate_text(step.output_text.trim(), 220))
        );
    }

    for (i, call) in step.tool_calls.iter().enumerate() {
        println!(
            "tool_call[{}]: {} args={}",
            i + 1,
            call.tool_name,
            compact_json(&call.args_json, 220)
        );
    }

    for (i, result) in step.tool_results.iter().enumerate() {
        println!(
            "tool_result[{}]: call_id={} is_error={} output={}",
            i + 1,
            result.call_id,
            result.is_error,
            compact_json(&result.output_json, 260)
        );
    }
}

fn compact_json(value: &Value, max_chars: usize) -> String {
    one_line(&truncate_text(&value.to_string(), max_chars))
}

fn one_line(text: &str) -> String {
    text.replace('\n', "\\n")
}
