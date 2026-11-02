//! Streaming chat against a mocked OpenAI-compatible server (a tiny local HTTP server in a
//! thread), with a usage-tracking middleware and the offline guard.
//!
//! `cargo run -p harpia --example streaming_client --features client`

use futures_util::StreamExt;
use harpia::client::{OpenAiCompatible, UsageTracker, WithMiddleware};
use harpia::core::traits::{ChatRequest, Provider};
use harpia::core::{Message, NetworkPolicy};
use harpia::http::{HttpClient, HttpConfig};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::Arc;

fn mock_server() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten().take(1) {
            let mut stream = stream;
            let mut buf = [0u8; 8192];
            let _ = stream.read(&mut buf);
            let body = ["The harpy", " eagle", " hunts", " sloths."]
                .iter()
                .map(|t| format!("data: {{\"choices\":[{{\"delta\":{{\"content\":\"{t}\"}}}}]}}\n\n"))
                .collect::<String>()
                + "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":4}}\n\ndata: [DONE]\n\n";
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
        }
    });
    format!("http://{addr}/v1")
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> harpia::Result<()> {
    let base = mock_server();
    let req = ChatRequest {
        model: "local".into(),
        messages: vec![Message::user("What does a harpy eagle eat?")],
        ..Default::default()
    };

    // offline by default: refused before any connection
    let offline = Arc::new(HttpClient::new(HttpConfig::default())?);
    let err = OpenAiCompatible::new(offline, &base, None).chat(&req).await.unwrap_err();
    println!("offline: {err}");

    let http = Arc::new(HttpClient::new(HttpConfig::new(NetworkPolicy::Online))?);
    let tracker = Arc::new(UsageTracker::default());
    let provider = WithMiddleware::new(OpenAiCompatible::new(http, &base, None)).layer(tracker);
    let mut stream = provider.chat_stream(&req).await?;
    print!("streamed: ");
    while let Some(chunk) = stream.next().await {
        let c = chunk?;
        print!("{}", c.delta);
        if let Some(u) = c.usage {
            print!("\n[usage: {} in, {} out]", u.input_tokens, u.output_tokens);
        }
    }
    println!();
    Ok(())
}
