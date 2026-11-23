//! The Wayland side of the preedit: one thread, one connection, an input
//! method object only while a take runs.

use super::state::{Event, Request};
use super::{Ack, Backend, Cmd, Driver, Shared, Sink, SpawnError};
use std::io::{Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::Arc;
use wayland_client::globals::{registry_queue_init, GlobalListContents};
use wayland_client::protocol::{wl_registry, wl_seat::WlSeat};
use wayland_client::{delegate_noop, Connection, Dispatch, EventQueue, Proxy, QueueHandle, WEnum};
use wayland_protocols::wp::text_input::zv3::client::zwp_text_input_v3::{
    ContentHint, ContentPurpose,
};
use wayland_protocols_misc::zwp_input_method_v2::client::zwp_input_method_manager_v2::ZwpInputMethodManagerV2;
use wayland_protocols_misc::zwp_input_method_v2::client::zwp_input_method_v2::{
    self, ZwpInputMethodV2,
};

struct App {
    driver: Driver,
    sink: WlSink,
}

struct WlSink {
    manager: ZwpInputMethodManagerV2,
    seat: WlSeat,
    qh: QueueHandle<App>,
    im: Option<ZwpInputMethodV2>,
}

impl Sink for WlSink {
    fn bind(&mut self) -> bool {
        self.im = Some(self.manager.get_input_method(&self.seat, &self.qh, ()));
        true
    }

    fn unbind(&mut self) {
        if let Some(im) = self.im.take() {
            im.destroy();
        }
    }

    fn send(&mut self, requests: Vec<Request>) {
        let Some(im) = self.im.as_ref() else {
            return;
        };
        for request in requests {
            match request {
                Request::Preedit(text) => {
                    let end = text.len() as i32;
                    im.set_preedit_string(text, end, end);
                }
                Request::CommitString(text) => im.commit_string(text),
                Request::Commit(serial) => im.commit(serial),
            }
        }
    }
}

impl Dispatch<ZwpInputMethodV2, ()> for App {
    fn event(
        app: &mut Self,
        proxy: &ZwpInputMethodV2,
        event: zwp_input_method_v2::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        // Events still queued for the object of an earlier take.
        if app.sink.im.as_ref().map(Proxy::id) != Some(proxy.id()) {
            return;
        }
        let event = match event {
            zwp_input_method_v2::Event::Activate => Event::Activate,
            zwp_input_method_v2::Event::Deactivate => Event::Deactivate,
            zwp_input_method_v2::Event::ContentType { hint, purpose } => {
                let secret_purpose = matches!(
                    purpose,
                    WEnum::Value(ContentPurpose::Password | ContentPurpose::Pin)
                );
                let secret_hint = matches!(
                    hint,
                    WEnum::Value(h) if h.intersects(ContentHint::SensitiveData | ContentHint::HiddenText)
                );
                Event::ContentHidden(secret_purpose || secret_hint)
            }
            zwp_input_method_v2::Event::Done => Event::Done,
            zwp_input_method_v2::Event::Unavailable => Event::Unavailable,
            _ => return,
        };
        app.driver.event(event, &mut app.sink);
    }
}

impl Dispatch<wl_registry::WlRegistry, GlobalListContents> for App {
    fn event(
        _: &mut Self,
        _: &wl_registry::WlRegistry,
        _: wl_registry::Event,
        _: &GlobalListContents,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

delegate_noop!(App: ignore WlSeat);
delegate_noop!(App: ZwpInputMethodManagerV2);

/// Connect, check the compositor speaks input-method-v2, and start the thread.
pub fn spawn() -> Result<Backend, SpawnError> {
    let conn = Connection::connect_to_env().map_err(|e| {
        tracing::warn!("Dictation preedit: no Wayland connection ({}); typing instead", e);
        SpawnError::Failed
    })?;
    let (globals, queue) = registry_queue_init::<App>(&conn).map_err(|e| {
        tracing::warn!("Dictation preedit: Wayland registry failed ({}); typing instead", e);
        SpawnError::Failed
    })?;
    let qh = queue.handle();
    let manager: ZwpInputMethodManagerV2 = globals.bind(&qh, 1..=1, ()).map_err(|_| {
        tracing::info!(
            "Compositor lacks zwp_input_method_manager_v2; dictation preedit stays off \
             (text is typed as before)"
        );
        SpawnError::Unsupported
    })?;
    let seat: WlSeat = globals.bind(&qh, 1..=1, ()).map_err(|_| {
        tracing::info!("Compositor has no wl_seat; dictation preedit stays off");
        SpawnError::Unsupported
    })?;

    let (wake_rx, wake_tx) = UnixStream::pair().map_err(|_| SpawnError::Failed)?;
    wake_rx.set_nonblocking(true).map_err(|_| SpawnError::Failed)?;
    wake_tx.set_nonblocking(true).map_err(|_| SpawnError::Failed)?;

    let shared = Arc::new(Shared::default());
    shared.alive.store(true, Ordering::Release);
    let (tx, rx) = std::sync::mpsc::channel();
    let app = App {
        driver: Driver::new(shared.clone()),
        sink: WlSink {
            manager,
            seat,
            qh,
            im: None,
        },
    };
    let thread_shared = shared.clone();
    std::thread::Builder::new()
        .name("voxtype-preedit".to_string())
        .spawn(move || {
            if let Err(e) = run(queue, app, rx, wake_rx) {
                tracing::warn!("Dictation preedit: Wayland connection lost ({})", e);
            }
            thread_shared.active.store(false, Ordering::Release);
            thread_shared.alive.store(false, Ordering::Release);
        })
        .map_err(|_| SpawnError::Failed)?;

    let wake = move || {
        // A full socket buffer already guarantees a wakeup.
        let _ = (&wake_tx).write(&[1]);
    };
    Ok(Backend::new(tx, Arc::new(wake), shared))
}

fn run(
    mut queue: EventQueue<App>,
    mut app: App,
    rx: Receiver<Cmd>,
    mut wake_rx: UnixStream,
) -> Result<(), Box<dyn std::error::Error>> {
    loop {
        let mut cmds = Vec::new();
        let mut disconnected = false;
        loop {
            match rx.try_recv() {
                Ok(cmd) => cmds.push(cmd),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }
        // Partials arrive faster than they are worth drawing: of a run of
        // updates only the newest goes out.
        let mut cmds = cmds.into_iter().peekable();
        while let Some(cmd) = cmds.next() {
            if matches!(cmd, Cmd::Set(_)) && matches!(cmds.peek(), Some(Cmd::Set(_))) {
                continue;
            }
            match app.driver.command(cmd, &mut app.sink) {
                Ack::None => {}
                Ack::Cleared(reply) => {
                    queue.roundtrip(&mut app)?;
                    let _ = reply.send(());
                }
                Ack::Committed(reply, outcome) => {
                    queue.roundtrip(&mut app)?;
                    let _ = reply.send(outcome);
                }
            }
        }
        if disconnected {
            app.driver.shutdown(&mut app.sink);
            queue.roundtrip(&mut app)?;
            return Ok(());
        }

        queue.dispatch_pending(&mut app)?;
        queue.flush()?;
        let Some(guard) = queue.prepare_read() else {
            continue;
        };
        let mut fds = [
            libc::pollfd {
                fd: guard.connection_fd().as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            },
            libc::pollfd {
                fd: wake_rx.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            },
        ];
        // SAFETY: `fds` is a valid array of two pollfd for the call's duration.
        let n = unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as libc::nfds_t, -1) };
        if n < 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(err.into());
        }
        if fds[0].revents != 0 {
            match guard.read() {
                Ok(_) => {}
                Err(wayland_client::backend::WaylandError::Io(e))
                    if e.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(e) => return Err(e.into()),
            }
        } else {
            drop(guard);
        }
        if fds[1].revents != 0 {
            let mut buf = [0u8; 64];
            while matches!(wake_rx.read(&mut buf), Ok(n) if n > 0) {}
        }
    }
}
