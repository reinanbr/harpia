//! Test helpers: prove that code under test opens no connections.

use std::net::{SocketAddr, TcpListener};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread::JoinHandle;

/// A local TCP listener that counts every accepted connection.
///
/// ```
/// use harpia_http::testing::ConnectionCounter;
/// let counter = ConnectionCounter::start();
/// let client = harpia_http::blocking::HttpClient::new(Default::default()).unwrap(); // offline
/// assert!(client.send(harpia_http::Request::get(counter.url())).is_err());
/// counter.assert_no_connections();
/// ```
pub struct ConnectionCounter {
    addr: SocketAddr,
    count: Arc<AtomicUsize>,
    _handle: JoinHandle<()>,
}

impl ConnectionCounter {
    /// Start listening on an ephemeral localhost port.
    pub fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind localhost");
        let addr = listener.local_addr().expect("local addr");
        let count = Arc::new(AtomicUsize::new(0));
        let c = count.clone();
        let handle = std::thread::spawn(move || {
            for stream in listener.incoming() {
                if stream.is_ok() {
                    c.fetch_add(1, Ordering::SeqCst);
                }
            }
        });
        Self { addr, count, _handle: handle }
    }

    /// `http://` URL of the listener.
    pub fn url(&self) -> String {
        format!("http://{}/", self.addr)
    }

    /// Connections accepted so far.
    pub fn connections(&self) -> usize {
        self.count.load(Ordering::SeqCst)
    }

    /// Panic if any connection was accepted.
    pub fn assert_no_connections(&self) {
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert_eq!(self.connections(), 0, "network access happened under an offline policy");
    }
}
