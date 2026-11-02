//! Common data types: chat messages, training samples, documents, scores, usage and cost.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

/// Free-form metadata attached to samples and documents.
pub type Metadata = BTreeMap<String, serde_json::Value>;

/// Author of a chat message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System / developer instructions.
    System,
    /// End user.
    User,
    /// Model output.
    Assistant,
    /// Tool result.
    Tool,
}

impl Role {
    /// Lowercase name as used by chat templates (`"system"`, `"user"`, ...).
    pub fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }

    /// Parse common role spellings, including ShareGPT's `human`/`gpt`.
    ///
    /// ```
    /// use harpia_core::Role;
    /// assert_eq!(Role::parse("human"), Some(Role::User));
    /// assert_eq!(Role::parse("gpt"), Some(Role::Assistant));
    /// ```
    pub fn parse(s: &str) -> Option<Role> {
        match s.trim().to_ascii_lowercase().as_str() {
            "system" | "developer" => Some(Role::System),
            "user" | "human" | "prompter" => Some(Role::User),
            "assistant" | "gpt" | "model" | "bot" | "chatgpt" => Some(Role::Assistant),
            "tool" | "function" | "ipython" | "observation" => Some(Role::Tool),
            _ => None,
        }
    }
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A tool/function call requested by the assistant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Call id (provider-specific), if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Tool name.
    pub name: String,
    /// Arguments as JSON.
    #[serde(default)]
    pub arguments: serde_json::Value,
}

/// One chat message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// Author.
    pub role: Role,
    /// Text content.
    #[serde(default)]
    pub content: String,
    /// Tool calls made in this (assistant) message.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// For tool messages: the call this result answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Optional author name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl Message {
    /// Message with `role` and `content`.
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self { role, content: content.into(), tool_calls: Vec::new(), tool_call_id: None, name: None }
    }
    /// System message.
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(Role::System, content)
    }
    /// User message.
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(Role::User, content)
    }
    /// Assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(Role::Assistant, content)
    }
    /// Tool result message.
    pub fn tool(call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self { tool_call_id: Some(call_id.into()), ..Self::new(Role::Tool, content) }
    }
}

/// An ordered list of messages.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Conversation {
    /// Messages in order.
    pub messages: Vec<Message>,
}

impl Conversation {
    /// Empty conversation.
    pub fn new() -> Self {
        Self::default()
    }
    /// Builder-style push.
    pub fn with(mut self, m: Message) -> Self {
        self.messages.push(m);
        self
    }
    /// Push a message.
    pub fn push(&mut self, m: Message) {
        self.messages.push(m);
    }
    /// Last assistant message, if any.
    pub fn last_assistant(&self) -> Option<&Message> {
        self.messages.iter().rev().find(|m| m.role == Role::Assistant)
    }
}

impl From<Vec<Message>> for Conversation {
    fn from(messages: Vec<Message>) -> Self {
        Self { messages }
    }
}

/// A chat-formatted training/eval sample.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ChatSample {
    /// The conversation.
    pub messages: Vec<Message>,
    /// Extra fields (source, license, provenance...).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: Metadata,
}

/// A preference pair for DPO-style training.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreferencePair {
    /// Prompt (conversation up to the answer).
    pub prompt: Vec<Message>,
    /// Preferred answer.
    pub chosen: String,
    /// Dispreferred answer.
    pub rejected: String,
    /// Extra fields.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: Metadata,
}

/// A text document with provenance.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Document {
    /// Stable identifier.
    pub id: String,
    /// Content.
    pub text: String,
    /// Where it came from (path, URL, dataset name).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// SPDX id or free-text license.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    /// Extra fields.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: Metadata,
}

impl Document {
    /// New document.
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self { id: id.into(), text: text.into(), ..Self::default() }
    }
}

/// A class label.
pub type Label = String;

/// Label → probability, kept in model label order.
///
/// ```
/// use harpia_core::Scores;
/// let s = Scores::new(vec![("supports".into(), 0.7), ("neutral".into(), 0.2), ("contradicts".into(), 0.1)]);
/// assert_eq!(s.argmax().unwrap().0, "supports");
/// assert_eq!(s.get("neutral"), Some(0.2));
/// ```
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Scores(pub Vec<(Label, f32)>);

impl Scores {
    /// From label/probability pairs.
    pub fn new(pairs: Vec<(Label, f32)>) -> Self {
        Self(pairs)
    }
    /// Probability of `label`.
    pub fn get(&self, label: &str) -> Option<f32> {
        self.0.iter().find(|(l, _)| l == label).map(|(_, p)| *p)
    }
    /// Highest-scoring label.
    pub fn argmax(&self) -> Option<(&str, f32)> {
        self.0.iter().max_by(|a, b| a.1.total_cmp(&b.1)).map(|(l, p)| (l.as_str(), *p))
    }
    /// Iterate over pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, f32)> {
        self.0.iter().map(|(l, p)| (l.as_str(), *p))
    }
}

/// Token accounting for one or more LLM calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Prompt/input tokens.
    pub input_tokens: u64,
    /// Completion/output tokens.
    pub output_tokens: u64,
    /// Input tokens served from a provider cache, when reported.
    #[serde(default)]
    pub cached_input_tokens: u64,
}

impl TokenUsage {
    /// Total tokens.
    pub fn total(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }
}

impl std::ops::Add for TokenUsage {
    type Output = TokenUsage;
    fn add(self, o: TokenUsage) -> TokenUsage {
        TokenUsage {
            input_tokens: self.input_tokens + o.input_tokens,
            output_tokens: self.output_tokens + o.output_tokens,
            cached_input_tokens: self.cached_input_tokens + o.cached_input_tokens,
        }
    }
}

impl std::ops::AddAssign for TokenUsage {
    fn add_assign(&mut self, o: TokenUsage) {
        *self = *self + o;
    }
}

/// A monetary cost.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Cost {
    /// Amount.
    pub amount: f64,
    /// ISO currency code (e.g. `"USD"`).
    pub currency: String,
}

impl fmt::Display for Cost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.6} {}", self.amount, self.currency)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_serde_roundtrip() {
        let m = Message::assistant("hi");
        let json = serde_json::to_string(&m).unwrap();
        assert_eq!(json, r#"{"role":"assistant","content":"hi"}"#);
        let back: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn usage_adds() {
        let mut u = TokenUsage { input_tokens: 3, output_tokens: 2, cached_input_tokens: 0 };
        u += TokenUsage { input_tokens: 1, output_tokens: 1, cached_input_tokens: 1 };
        assert_eq!(u.total(), 7);
    }
}
