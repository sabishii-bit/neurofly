//! A trained neurofly controller from Rust.
//!
//! Spawns `neurofly-core serve <artifact>` and talks JSON lines to it (see
//! `artifact/SPEC.md`). The runtime must be installed in a Python on the PATH
//! (`pip install neurofly-core`), or give [`Client::with_python`] an interpreter.
//!
//! ```no_run
//! use neurofly::{Client, Step};
//! let mut fly = Client::spawn("artifacts/myapp").unwrap();
//! let frame = vec![0u8; 320 * 240 * 3];
//! let r = fly.step(&Step::rgb(&frame, 320, 240)).unwrap();
//! println!("{:?} dx={} spikes={}", r.keys, r.dx, r.spikes);
//! fly.close().unwrap();
//! ```

use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

#[derive(Debug, Deserialize)]
pub struct Info {
    pub name: String,
    pub n_neurons: usize,
    pub n_features: usize,
    pub n_actions: usize,
    pub controls: Vec<String>,
    pub brain_ms: f64,
    pub has_policy: bool,
    pub has_audition: bool,
    pub sample_rate: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct StepResult {
    pub t: u64,
    pub spikes: u64,
    #[serde(default)]
    pub action: Vec<f32>,
    #[serde(default)]
    pub held: Vec<String>,
    #[serde(default)]
    pub keys: Vec<String>,
    #[serde(default)]
    pub buttons: Vec<String>,
    #[serde(default)]
    pub dx: f64,
    #[serde(default)]
    pub dy: f64,
    #[serde(default)]
    pub scroll: f64,
    #[serde(default)]
    pub features: Vec<f32>,
}

/// One step's input. Build with [`Step::rgb`] or [`Step::encoded`], then chain
/// [`Step::audio`] and [`Step::reward`].
#[derive(Serialize)]
pub struct Step {
    op: &'static str,
    frame: String,
    width: u32,
    height: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    audio: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sample_rate: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    channels: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reward: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    observe_only: Option<bool>,
}

impl Step {
    /// Raw RGB bytes, row-major, three per pixel.
    pub fn rgb(frame: &[u8], width: u32, height: u32) -> Step {
        Step { op: "step", frame: B64.encode(frame), width, height, format: None, audio: None,
               sample_rate: None, channels: None, reward: None, observe_only: None }
    }

    /// An encoded image (`"png"` or `"jpeg"`).
    pub fn encoded(bytes: &[u8], format: &'static str) -> Step {
        let mut s = Step::rgb(bytes, 0, 0);
        s.format = Some(format);
        s
    }

    /// Samples since the last step, in [-1, 1], interleaved if `channels > 1`.
    pub fn audio(mut self, samples: &[f32], sample_rate: u32, channels: u32) -> Step {
        let mut bytes = Vec::with_capacity(samples.len() * 4);
        for x in samples {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        self.audio = Some(B64.encode(&bytes));
        self.sample_rate = Some(sample_rate);
        self.channels = Some(channels);
        self
    }

    pub fn reward(mut self, reward: f64) -> Step {
        self.reward = Some(reward);
        self
    }

    /// Return the feature vector instead of controls.
    pub fn observe_only(mut self) -> Step {
        self.observe_only = Some(true);
        self
    }
}

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Json(serde_json::Error),
    Runtime(String),
    Exited,
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self { Error::Io(e) }
}
impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self { Error::Json(e) }
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io: {e}"),
            Error::Json(e) => write!(f, "json: {e}"),
            Error::Runtime(m) => write!(f, "neurofly-core: {m}"),
            Error::Exited => write!(f, "neurofly-core exited"),
        }
    }
}
impl std::error::Error for Error {}

#[derive(Deserialize)]
struct Envelope {
    ok: bool,
    #[serde(default)]
    error: Option<String>,
}

pub struct Client {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    pub info: Info,
}

impl Client {
    /// Start `neurofly-core serve <artifact>`.
    pub fn spawn(artifact: &str) -> Result<Client, Error> {
        Client::start(Command::new("neurofly-core").args(["serve", artifact]))
    }

    /// Start `<python> -m neurofly_core serve <artifact>`.
    pub fn with_python(python: &str, artifact: &str) -> Result<Client, Error> {
        Client::start(Command::new(python).args(["-m", "neurofly_core", "serve", artifact]))
    }

    fn start(cmd: &mut Command) -> Result<Client, Error> {
        let mut child = cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit()).spawn()?;
        let stdin = child.stdin.take().ok_or(Error::Exited)?;
        let stdout = BufReader::new(child.stdout.take().ok_or(Error::Exited)?);
        let mut c = Client { child, stdin, stdout, info: Info {
            name: String::new(), n_neurons: 0, n_features: 0, n_actions: 0, controls: vec![],
            brain_ms: 0.0, has_policy: false, has_audition: false, sample_rate: None } };
        let ready = c.read_line()?;                   // the ready line carries the info
        c.info = serde_json::from_str(&ready)?;
        Ok(c)
    }

    fn read_line(&mut self) -> Result<String, Error> {
        let mut line = String::new();
        if self.stdout.read_line(&mut line)? == 0 {
            return Err(Error::Exited);
        }
        let env: Envelope = serde_json::from_str(&line)?;
        if !env.ok {
            return Err(Error::Runtime(env.error.unwrap_or_default()));
        }
        Ok(line)
    }

    fn call<T: for<'de> Deserialize<'de>>(&mut self, req: &impl Serialize) -> Result<T, Error> {
        let mut s = serde_json::to_string(req)?;
        s.push('\n');
        self.stdin.write_all(s.as_bytes())?;
        self.stdin.flush()?;
        let line = self.read_line()?;
        Ok(serde_json::from_str(&line)?)
    }

    /// Start a fresh episode.
    pub fn reset(&mut self) -> Result<(), Error> {
        let _: serde_json::Value = self.call(&serde_json::json!({"op": "reset"}))?;
        Ok(())
    }

    /// One step: a frame (and sound) in, controls out.
    pub fn step(&mut self, step: &Step) -> Result<StepResult, Error> {
        self.call(step)
    }

    /// Like `step`, but returns the feature vector (`features`) instead of controls.
    pub fn observe(&mut self, step: &Step) -> Result<StepResult, Error> {
        let mut s = serde_json::to_value(step)?;
        s["op"] = serde_json::Value::from("observe");
        self.call(&s)
    }

    /// Install a linear policy you trained: `w` is `actions x features`, `b` is `actions`.
    pub fn set_policy_linear(&mut self, w: &[Vec<f64>], b: &[f64]) -> Result<(), Error> {
        let _: serde_json::Value = self.call(&serde_json::json!({
            "op": "set_policy", "type": "linear", "W": w, "b": b }))?;
        Ok(())
    }

    /// Write the runtime's current model (with the installed policy) as an artifact.
    pub fn save(&mut self, dir: &str, name: Option<&str>) -> Result<String, Error> {
        #[derive(Deserialize)]
        struct Saved { path: String }
        let r: Saved = self.call(&serde_json::json!({"op": "save", "path": dir, "name": name}))?;
        Ok(r.path)
    }

    /// Stop the runtime; every key and button it held is released.
    pub fn close(mut self) -> Result<(), Error> {
        let _ = self.call::<serde_json::Value>(&serde_json::json!({"op": "close"}));
        let _ = self.child.wait();
        Ok(())
    }
}
