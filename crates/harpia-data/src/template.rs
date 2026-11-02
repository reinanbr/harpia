//! Chat templates rendered with the model's own Jinja `chat_template` (via `minijinja`, with
//! Python-compatible string/list methods), plus assistant spans for loss masks.

use harpia_core::span::Span;
use harpia_core::traits::{ChatTemplate, RenderedChat};
use harpia_core::types::{Message, Role};
use harpia_core::{HarpiaError, Result};
use minijinja::{Environment, Error as JinjaError, ErrorKind, context};
use serde_json::Value;
use std::path::Path;

/// ChatML (Qwen, many fine-tunes).
pub const CHATML: &str = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

/// Llama 3 instruct.
pub const LLAMA3: &str = "{{ bos_token }}{% for message in messages %}{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}";

/// Plain `Role: content` lines, for base models.
pub const PLAIN: &str = "{% for message in messages %}{{ message['role'] | capitalize + ': ' + message['content'] + '\\n' }}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant: ' }}{% endif %}";

/// A Jinja chat template with its special tokens.
///
/// ```
/// use harpia_core::traits::ChatTemplate;
/// use harpia_core::types::Message;
/// use harpia_data::template::{JinjaChatTemplate, CHATML};
///
/// let t = JinjaChatTemplate::new(CHATML, "", "<|im_end|>").unwrap();
/// let msgs = [Message::user("Oi"), Message::assistant("Olá!")];
/// let r = t.render_with_spans(&msgs).unwrap();
/// assert_eq!(r.assistant_spans.len(), 1);
/// assert_eq!(r.assistant_spans[0].slice(&r.text), "Olá!<|im_end|>\n");
/// ```
#[derive(Debug)]
pub struct JinjaChatTemplate {
    env: Environment<'static>,
    bos: String,
    eos: String,
}

fn raise_exception(msg: String) -> std::result::Result<String, JinjaError> {
    Err(JinjaError::new(ErrorKind::InvalidOperation, msg))
}

/// `strftime_now(fmt)` used by some templates (Llama 3.1+); supports `%d %m %Y %b %B`.
fn strftime_now(fmt: String) -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let days = (secs / 86_400) as i64;
    // civil-from-days (Howard Hinnant)
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    const MONTHS: [&str; 12] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    let name = MONTHS[(m - 1) as usize];
    fmt.replace("%d", &format!("{d:02}"))
        .replace("%m", &format!("{m:02}"))
        .replace("%Y", &y.to_string())
        .replace("%B", name)
        .replace("%b", &name[..3])
}

impl JinjaChatTemplate {
    /// Compile `template` with the given BOS/EOS token strings.
    pub fn new(template: &str, bos_token: &str, eos_token: &str) -> Result<Self> {
        let mut env = Environment::new();
        // Hugging Face renders chat templates with trim_blocks and lstrip_blocks enabled
        env.set_syntax(
            minijinja::syntax::SyntaxConfig::builder()
                .trim_blocks(true)
                .lstrip_blocks(true)
                .build()
                .map_err(|e| HarpiaError::Template(e.to_string()))?,
        );
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", raise_exception);
        env.add_function("strftime_now", strftime_now);
        env.add_template_owned("chat", template.to_string()).map_err(|e| HarpiaError::Template(e.to_string()))?;
        Ok(Self { env, bos: bos_token.to_string(), eos: eos_token.to_string() })
    }

    /// Load `chat_template`, `bos_token` and `eos_token` from a Hugging Face
    /// `tokenizer_config.json`. Template lists use the entry named `default`.
    pub fn from_tokenizer_config(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|e| HarpiaError::io(path, e))?;
        let cfg: Value = serde_json::from_str(&text)?;
        let template = match cfg.get("chat_template") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(list)) => list
                .iter()
                .find(|t| t.get("name").and_then(Value::as_str) == Some("default"))
                .or(list.first())
                .and_then(|t| t.get("template").and_then(Value::as_str))
                .map(String::from)
                .ok_or_else(|| HarpiaError::Template("empty chat_template list".into()))?,
            _ => return Err(HarpiaError::Template(format!("{}: no chat_template", path.display()))),
        };
        let tok = |k: &str| match cfg.get(k) {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Object(o)) => o.get("content").and_then(Value::as_str).unwrap_or("").to_string(),
            _ => String::new(),
        };
        Self::new(&template, &tok("bos_token"), &tok("eos_token"))
    }

    fn render_raw(&self, messages: &[Message], add_generation_prompt: bool) -> Result<String> {
        let tmpl = self.env.get_template("chat").map_err(|e| HarpiaError::Template(e.to_string()))?;
        tmpl.render(context! {
            messages => minijinja::Value::from(minijinja::value::Serde(messages)),
            add_generation_prompt => add_generation_prompt,
            bos_token => &self.bos,
            eos_token => &self.eos,
        })
        .map_err(|e| HarpiaError::Template(format!("{e:#}")))
    }
}

impl ChatTemplate for JinjaChatTemplate {
    fn render(&self, messages: &[Message], add_generation_prompt: bool) -> Result<String> {
        self.render_raw(messages, add_generation_prompt)
    }

    /// Assistant spans are found by rendering growing prefixes of the conversation: the text
    /// added by assistant turn `i` (relative to the prompt rendered with a generation header)
    /// is that turn's span. Fails if the template is not prefix-stable.
    fn render_with_spans(&self, messages: &[Message]) -> Result<RenderedChat> {
        let full = self.render_raw(messages, false)?;
        let mut spans = Vec::new();
        for (i, m) in messages.iter().enumerate() {
            if m.role != Role::Assistant {
                continue;
            }
            let before = self.render_raw(&messages[..i], true)?;
            let upto = self.render_raw(&messages[..=i], false)?;
            if !upto.starts_with(&before) || !full.starts_with(&upto) {
                return Err(HarpiaError::Template(
                    "chat template is not prefix-stable; cannot locate assistant spans (render each turn separately)"
                        .into(),
                ));
            }
            if upto.len() > before.len() {
                spans.push(Span::new(before.len(), upto.len()));
            }
        }
        Ok(RenderedChat { text: full, assistant_spans: spans })
    }
}

/// Token-level loss mask: `true` for tokens whose byte span overlaps an assistant span.
pub fn loss_mask(token_offsets: &[Span], assistant_spans: &[Span]) -> Vec<bool> {
    token_offsets
        .iter()
        .map(|t| !t.is_empty() && assistant_spans.iter().any(|a| t.start < a.end && a.start < t.end))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama3_and_generation_prompt() {
        let t = JinjaChatTemplate::new(LLAMA3, "<|begin_of_text|>", "<|eot_id|>").unwrap();
        let p = t.render(&[Message::system("Be brief."), Message::user("Hi")], true).unwrap();
        assert!(p.starts_with("<|begin_of_text|><|start_header_id|>system"));
        assert!(p.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn multi_turn_spans() {
        let t = JinjaChatTemplate::new(CHATML, "", "").unwrap();
        let msgs = [Message::user("a"), Message::assistant("b"), Message::user("c"), Message::assistant("d")];
        let r = t.render_with_spans(&msgs).unwrap();
        let parts: Vec<&str> = r.assistant_spans.iter().map(|s| s.slice(&r.text)).collect();
        assert_eq!(parts, ["b<|im_end|>\n", "d<|im_end|>\n"]);
        let mask = loss_mask(&[Span::new(0, 3), r.assistant_spans[0], Span::new(0, 0)], &r.assistant_spans);
        assert_eq!(mask, [false, true, false]);
    }

    #[test]
    fn python_methods_and_errors() {
        let t = JinjaChatTemplate::new("{% if messages[0]['role'] != 'user' %}{{ raise_exception('first must be user') }}{% endif %}{{ messages[0]['content'].strip().upper() }}", "", "").unwrap();
        assert_eq!(t.render(&[Message::user("  oi ")], false).unwrap(), "OI");
        let err = t.render(&[Message::system("x")], false).unwrap_err().to_string();
        assert!(err.contains("first must be user"), "{err}");
        assert!(strftime_now("%d %b %Y".into()).len() >= 10);
    }
}
