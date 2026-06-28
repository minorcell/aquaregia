mod tools;

use std::collections::HashMap;
use std::io::{self, Stdout};
use std::sync::Arc;
use std::time::Duration;

use aquaregia::{Agent, AgentStreamEvent, Message, StreamEvent};
use crossterm::event::{self, Event as CrosstermEvent, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use futures_util::StreamExt;
use ratatui::backend::CrosstermBackend;
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use serde_json::Value;
use tokio::sync::mpsc;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

const DEFAULT_BASE_URL: &str = "https://api.deepseek.com";
const DEFAULT_MODEL: &str = "deepseek-v4-pro";
const SYSTEM_PROMPT: &str = r#"
You are a local terminal code agent.

Rules:
1. Read files before editing them.
2. Make the smallest change that solves the user's task.
3. Prefer targeted edits over full-file writes.
4. Explain what changed and which files were touched.
5. Ask for clarification when the task is ambiguous.
"#;

#[derive(Debug)]
struct ChatLine {
    role: String,
    content: String,
}

struct App {
    input: String,
    lines: Vec<ChatLine>,
    history: Vec<Message>,
    running: bool,
    assistant_by_step: HashMap<u32, usize>,
    tool_by_call: HashMap<String, usize>,
    status: String,
}

enum WorkerEvent {
    AssistantDelta {
        step: u32,
        text: String,
    },
    ToolStart {
        step: u32,
        call_id: String,
        name: String,
        args: String,
    },
    ToolFinish {
        step: u32,
        call_id: String,
        name: String,
        args: String,
        output: String,
        is_error: bool,
    },
    Done {
        transcript: Vec<Message>,
        summary: String,
    },
    Error(String),
}

type Terminal = ratatui::Terminal<CrosstermBackend<Stdout>>;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    let base_url =
        std::env::var("DEEPSEEK_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
    let client = aquaregia::providers::openai_compatible::Client::builder()
        .base_url(base_url)
        .api_key_from_env("DEEPSEEK_API_KEY")
        .build()?;
    let agent = Arc::new(
        client
            .agent(model.clone())
            .instructions(SYSTEM_PROMPT)
            .tools([tools::bash()])
            .tools([tools::read()])
            .tools([tools::write()])
            .tools([tools::edit()])
            .max_steps(12)
            .temperature(0.2)
            .max_output_tokens(1_600)
            .build()?,
    );

    let mut terminal = setup_terminal()?;
    let result = run_app(&mut terminal, agent, model).await;
    restore_terminal(&mut terminal)?;
    result
}

async fn run_app(
    terminal: &mut Terminal,
    agent: Arc<Agent>,
    model: String,
) -> Result<(), Box<dyn std::error::Error>> {
    let (tx, mut rx) = mpsc::unbounded_channel();
    let mut app = App {
        input: String::new(),
        lines: vec![ChatLine {
            role: "System".to_string(),
            content: format!("aquaregia claude_code example | model=deepseek/{model} | Esc exits"),
        }],
        history: Vec::new(),
        running: false,
        assistant_by_step: HashMap::new(),
        tool_by_call: HashMap::new(),
        status: "Ready".to_string(),
    };

    loop {
        while let Ok(event) = rx.try_recv() {
            app.apply(event);
        }

        terminal.draw(|frame| draw(frame, &app))?;

        if event::poll(Duration::from_millis(80))?
            && let CrosstermEvent::Key(key) = event::read()?
            && key.kind == KeyEventKind::Press
        {
            match key.code {
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => break,
                KeyCode::Esc => break,
                KeyCode::Enter if !app.running => {
                    let prompt = app.input.trim().to_string();
                    if !prompt.is_empty() {
                        app.submit(prompt, Arc::clone(&agent), tx.clone());
                    }
                }
                KeyCode::Char(ch) if !app.running => app.input.push(ch),
                KeyCode::Backspace if !app.running => {
                    app.input.pop();
                }
                _ => {}
            }
        }
    }

    Ok(())
}

impl App {
    fn submit(
        &mut self,
        prompt: String,
        agent: Arc<Agent>,
        tx: mpsc::UnboundedSender<WorkerEvent>,
    ) {
        self.input.clear();
        self.running = true;
        self.status = "Running agent".to_string();
        self.assistant_by_step.clear();
        self.tool_by_call.clear();
        self.lines.push(ChatLine {
            role: "You".to_string(),
            content: prompt.clone(),
        });

        let mut messages = self.history.clone();
        messages.push(Message::user_text(prompt));

        tokio::spawn(async move {
            run_agent(agent, messages, tx).await;
        });
    }

    fn apply(&mut self, event: WorkerEvent) {
        match event {
            WorkerEvent::AssistantDelta { step, text } => {
                let index = self.assistant_line_for_step(step);
                self.lines[index].content.push_str(&text);
            }
            WorkerEvent::ToolStart {
                step,
                call_id,
                name,
                args,
            } => {
                self.status = format!("Tool running: {name}");
                self.lines.push(ChatLine {
                    role: format!("Tool {step}"),
                    content: format!("{name}({args})\n..."),
                });
                self.tool_by_call.insert(call_id, self.lines.len() - 1);
            }
            WorkerEvent::ToolFinish {
                step,
                call_id,
                name,
                args,
                output,
                is_error,
            } => {
                self.status = if is_error {
                    format!("Tool failed: {name}")
                } else {
                    format!("Tool finished: {name}")
                };
                let content = if is_error {
                    format!("{name}({args})\nerror: {output}")
                } else {
                    format!("{name}({args})\n=> {output}")
                };
                if let Some(index) = self.tool_by_call.get(&call_id).copied() {
                    self.lines[index].content = content;
                } else {
                    self.lines.push(ChatLine {
                        role: format!("Tool {step}"),
                        content,
                    });
                }
            }
            WorkerEvent::Done {
                transcript,
                summary,
            } => {
                self.history = transcript;
                self.running = false;
                self.assistant_by_step.clear();
                self.tool_by_call.clear();
                self.status = summary;
            }
            WorkerEvent::Error(message) => {
                self.running = false;
                self.assistant_by_step.clear();
                self.tool_by_call.clear();
                self.status = "Error".to_string();
                self.lines.push(ChatLine {
                    role: "Error".to_string(),
                    content: message,
                });
            }
        }
    }

    fn assistant_line_for_step(&mut self, step: u32) -> usize {
        if let Some(index) = self.assistant_by_step.get(&step).copied() {
            return index;
        }

        self.lines.push(ChatLine {
            role: format!("Assistant {step}"),
            content: String::new(),
        });
        let index = self.lines.len() - 1;
        self.assistant_by_step.insert(step, index);
        index
    }
}

async fn run_agent(
    agent: Arc<Agent>,
    messages: Vec<Message>,
    tx: mpsc::UnboundedSender<WorkerEvent>,
) {
    let stream = match agent.stream_messages(messages).await {
        Ok(stream) => stream,
        Err(err) => {
            let _ = tx.send(WorkerEvent::Error(err.to_string()));
            return;
        }
    };

    tokio::pin!(stream);
    while let Some(event) = stream.next().await {
        match event {
            Ok(AgentStreamEvent::Model {
                step,
                event: StreamEvent::TextDelta { text },
                ..
            }) => {
                let _ = tx.send(WorkerEvent::AssistantDelta { step, text });
            }
            Ok(AgentStreamEvent::ToolCallStart { event }) => {
                let _ = tx.send(WorkerEvent::ToolStart {
                    step: event.step,
                    call_id: event.tool_call.call_id,
                    name: event.tool_call.tool_name,
                    args: compact_json(&event.tool_call.args_json, 180),
                });
            }
            Ok(AgentStreamEvent::ToolCallFinish { event }) => {
                let output =
                    format_tool_result(&event.tool_result.output_json, event.tool_result.is_error);
                let _ = tx.send(WorkerEvent::ToolFinish {
                    step: event.step,
                    call_id: event.tool_call.call_id,
                    name: event.tool_call.tool_name,
                    args: compact_json(&event.tool_call.args_json, 180),
                    output,
                    is_error: event.tool_result.is_error,
                });
            }
            Ok(AgentStreamEvent::Done { output }) => {
                let summary = format!(
                    "Done | steps={} | tokens={}",
                    output.steps, output.usage_total.total_tokens
                );
                let _ = tx.send(WorkerEvent::Done {
                    transcript: output.transcript,
                    summary,
                });
            }
            Ok(_) => {}
            Err(err) => {
                let _ = tx.send(WorkerEvent::Error(err.to_string()));
                return;
            }
        }
    }
}

fn draw(frame: &mut Frame<'_>, app: &App) {
    let [messages_area, input_area, status_area] = Layout::vertical([
        Constraint::Min(8),
        Constraint::Length(3),
        Constraint::Length(1),
    ])
    .areas(frame.area());

    let content_width = messages_area.width.saturating_sub(2).max(1) as usize;
    let content_height = messages_area.height.saturating_sub(2).max(1) as usize;
    let conversation = conversation_rows(app, content_width, content_height);

    frame.render_widget(
        Paragraph::new(conversation)
            .block(Block::new().title("Conversation").borders(Borders::ALL)),
        messages_area,
    );

    frame.render_widget(
        Paragraph::new(app.input.as_str())
            .block(Block::new().title("Message").borders(Borders::ALL))
            .wrap(Wrap { trim: false }),
        input_area,
    );

    frame.render_widget(Paragraph::new(app.status.as_str()), status_area);

    if !app.running {
        let input_width = input_area.width.saturating_sub(2);
        let cursor_offset = app.input.width().min(input_width as usize) as u16;
        frame.set_cursor_position(Position::new(
            input_area.x + 1 + cursor_offset,
            input_area.y + 1,
        ));
    }
}

fn conversation_rows(app: &App, width: usize, height: usize) -> Vec<Line<'static>> {
    let mut rows = Vec::new();

    for line in &app.lines {
        rows.push(Line::from(Span::styled(
            line.role.clone(),
            Style::default().add_modifier(Modifier::BOLD),
        )));
        rows.extend(wrap_text(&line.content, width).into_iter().map(Line::from));
    }

    let start = rows.len().saturating_sub(height);
    rows.into_iter().skip(start).collect()
}

fn wrap_text(text: &str, max_width: usize) -> Vec<String> {
    let max_width = max_width.max(1);
    let mut rows = Vec::new();

    for source_line in text.split('\n') {
        if source_line.is_empty() {
            rows.push(String::new());
            continue;
        }

        let mut row = String::new();
        let mut row_width = 0usize;

        for ch in source_line.chars() {
            let ch_width = ch.width().unwrap_or(0);
            if row_width > 0 && row_width + ch_width > max_width {
                rows.push(row);
                row = String::new();
                row_width = 0;
            }

            row.push(ch);
            row_width += ch_width;
        }

        rows.push(row);
    }

    if rows.is_empty() {
        rows.push(String::new());
    }

    rows
}

fn setup_terminal() -> io::Result<Terminal> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    Terminal::new(CrosstermBackend::new(stdout))
}

fn restore_terminal(terminal: &mut Terminal) -> io::Result<()> {
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()
}

fn compact_json(value: &Value, max_chars: usize) -> String {
    truncate(&value.to_string().replace('\n', "\\n"), max_chars)
}

fn format_tool_result(value: &Value, is_error: bool) -> String {
    if is_error {
        return compact_json(value, 220);
    }

    if let Some(object) = value.as_object() {
        if let Some(output) = object.get("output").and_then(Value::as_str) {
            let exit_code = object.get("exit_code").and_then(Value::as_i64).unwrap_or(0);
            return format!("exit {exit_code}: {}", truncate_inline(output, 180));
        }

        if let Some(content) = object.get("content").and_then(Value::as_str) {
            let path = object.get("path").and_then(Value::as_str).unwrap_or("file");
            let start = object
                .get("line_start")
                .and_then(Value::as_u64)
                .unwrap_or(1);
            let end = object
                .get("line_end")
                .and_then(Value::as_u64)
                .unwrap_or(start);
            return format!(
                "{path} lines {start}-{end}: {}",
                truncate_inline(content, 180)
            );
        }

        if let Some(bytes) = object.get("bytes_written").and_then(Value::as_u64) {
            let path = object.get("path").and_then(Value::as_str).unwrap_or("file");
            return format!("{path}: wrote {bytes} bytes");
        }

        if object.get("replaced").and_then(Value::as_bool) == Some(true) {
            let path = object.get("path").and_then(Value::as_str).unwrap_or("file");
            return format!("{path}: replaced one match");
        }
    }

    compact_json(value, 220)
}

fn truncate_inline(text: &str, max_chars: usize) -> String {
    truncate(&text.replace('\n', "\\n"), max_chars)
}

fn truncate(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }

    let mut out = text.chars().take(max_chars).collect::<String>();
    out.push_str("...");
    out
}
