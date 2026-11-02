//! Behavior of the polite client against a local mock server.

use harpia_core::NetworkPolicy;
use harpia_http::testing::ConnectionCounter;
use harpia_http::{CacheConfig, HttpClient, HttpConfig, Request, parse_retry_after};
use std::time::Duration;
use wiremock::matchers::{method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn online() -> HttpConfig {
    HttpConfig::new(NetworkPolicy::Online).retries(3, Duration::from_millis(5))
}

#[tokio::test]
async fn caches_get_responses_ignoring_excluded_params() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/works"))
        .and(query_param("q", "harpia"))
        .respond_with(ResponseTemplate::new(200).set_body_string("{\"n\": 1}"))
        .expect(1)
        .mount(&server)
        .await;
    let dir = tempfile::tempdir().unwrap();
    let client = HttpClient::new(online().with_cache(CacheConfig::new(dir.path().join("c.redb")))).unwrap();
    let url = format!("{}/works", server.uri());
    let a = client.send(Request::get(&url).param("q", "harpia").param("mailto", "a@x.org")).await.unwrap();
    let b = client.send(Request::get(&url).param("mailto", "b@y.org").param("q", "harpia")).await.unwrap();
    assert!(!a.from_cache && b.from_cache);
    assert_eq!(b.json::<serde_json::Value>().unwrap()["n"], 1);
    assert_eq!(client.requests_sent(), 1);
}

#[tokio::test]
async fn retries_429_honoring_retry_after() {
    let server = MockServer::start().await;
    Mock::given(path("/limited"))
        .respond_with(ResponseTemplate::new(429).insert_header("retry-after", "0"))
        .up_to_n_times(2)
        .mount(&server)
        .await;
    Mock::given(path("/limited")).respond_with(ResponseTemplate::new(200).set_body_string("ok")).mount(&server).await;
    let client = HttpClient::new(online()).unwrap();
    let r = client.send(Request::get(format!("{}/limited", server.uri()))).await.unwrap();
    assert_eq!((r.status, r.text().as_str()), (200, "ok"));
    assert_eq!(client.requests_sent(), 3);
}

#[tokio::test]
async fn circuit_breaker_disables_failing_host() {
    let server = MockServer::start().await;
    Mock::given(path("/down")).respond_with(ResponseTemplate::new(503)).mount(&server).await;
    let mut cfg = online().retries(1, Duration::from_millis(1));
    cfg.failures_to_disable = 2;
    let client = HttpClient::new(cfg).unwrap();
    let url = format!("{}/down", server.uri());
    for _ in 0..2 {
        assert!(client.send(Request::get(&url)).await.is_err());
    }
    let sent = client.requests_sent();
    let err = client.send(Request::get(&url)).await.unwrap_err().to_string();
    assert!(err.contains("disabled"), "{err}");
    assert_eq!(client.requests_sent(), sent, "no request once disabled");
    let st = &client.status()[0];
    assert!(st.disabled && st.failures == 2 && st.last_error.as_deref() == Some("HTTP 503"));
}

#[tokio::test]
async fn non_retryable_statuses_are_returned() {
    let server = MockServer::start().await;
    Mock::given(path("/missing")).respond_with(ResponseTemplate::new(404)).expect(1).mount(&server).await;
    let client = HttpClient::new(online()).unwrap();
    assert_eq!(client.send(Request::get(format!("{}/missing", server.uri()))).await.unwrap().status, 404);
}

#[tokio::test]
async fn body_size_cap_and_capped_download() {
    let server = MockServer::start().await;
    Mock::given(path("/big"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(vec![b'x'; 4096]))
        .mount(&server)
        .await;
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = online().with_cache(CacheConfig::new(dir.path().join("c.redb")));
    cfg.max_body_bytes = 1024;
    let client = HttpClient::new(cfg).unwrap();
    let url = format!("{}/big", server.uri());
    assert!(client.send(Request::get(&url)).await.unwrap_err().to_string().contains("larger than"));
    assert!(client.download(&url, 100).await.is_err());
    assert_eq!(client.download(&url, 10_000).await.unwrap().len(), 4096);
    let before = client.requests_sent();
    assert_eq!(client.download(&url, 10_000).await.unwrap().len(), 4096);
    assert_eq!(client.requests_sent(), before, "second download served from disk");
}

#[tokio::test]
async fn streaming_body() {
    use futures_util::StreamExt;
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/sse"))
        .respond_with(ResponseTemplate::new(200).set_body_string("data: a\n\ndata: b\n\n"))
        .mount(&server)
        .await;
    let client = HttpClient::new(online()).unwrap();
    let mut r = client
        .send_streaming(Request::post_json(format!("{}/sse", server.uri()), &serde_json::json!({})))
        .await
        .unwrap();
    let mut all = Vec::new();
    while let Some(chunk) = r.body.next().await {
        all.extend_from_slice(&chunk.unwrap());
    }
    assert_eq!(r.status, 200);
    assert_eq!(String::from_utf8(all).unwrap(), "data: a\n\ndata: b\n\n");
}

#[test]
fn offline_opens_no_socket() {
    let counter = ConnectionCounter::start();
    let client = harpia_http::blocking::HttpClient::new(HttpConfig::default()).unwrap();
    let err = client.send(Request::get(counter.url())).unwrap_err();
    assert!(matches!(err, harpia_core::HarpiaError::NetworkDisabled { .. }));
    assert_eq!(client.inner().requests_refused(), 1);
    counter.assert_no_connections();
}

#[test]
fn retry_after_forms() {
    assert_eq!(parse_retry_after("3"), Some(Duration::from_secs(3)));
    assert_eq!(parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT"), Some(Duration::ZERO));
    assert_eq!(parse_retry_after("soon"), None);
}
