//! The native half of the Python package. Trainers stay in Python; everything
//! that touches masks, keys or the network stays here with the GIL released.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use fedtally_core::keys::Identity;
use fedtally_core::model::{Dtype, Layout, TensorSpec, TensorView, write_model};
use fedtally_core::report::{VerifyError, verify_log};
use fedtally_node::NodeError;
use fedtally_node::aggregator::run_aggregator;
use fedtally_node::helper::run_helper;
use fedtally_node::local::{GrantFile, LocalFederation, Options, StandingFile};
use fedtally_node::registry::RpcRegistry;
use fedtally_node::site::{
    RevocationFile, SiteConfig, SiteSummary, Submit, TrainContext, Trainer, run_site,
};
use pyo3::buffer::PyBuffer;
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

create_exception!(_native, FedtallyError, PyException);
create_exception!(_native, VerifyFailed, FedtallyError);

fn fail(e: impl std::fmt::Display) -> PyErr {
    FedtallyError::new_err(e.to_string())
}

fn parse_layout(tensors: Vec<(String, String, Vec<u64>)>) -> PyResult<Layout> {
    let tensors = tensors
        .into_iter()
        .map(|(name, dtype, shape)| {
            let dtype = match dtype.as_str() {
                "F32" => Dtype::F32,
                "F64" => Dtype::F64,
                other => return Err(PyValueError::new_err(format!("unsupported dtype {other}"))),
            };
            Ok(TensorSpec { name, dtype, shape })
        })
        .collect::<PyResult<_>>()?;
    Ok(Layout { tensors })
}

enum Buffer {
    F32(PyBuffer<f32>),
    F64(PyBuffer<f64>),
}

impl Buffer {
    fn get(array: &Bound<'_, PyAny>, dtype: Dtype) -> PyResult<Buffer> {
        let buffer = match dtype {
            Dtype::F32 => Buffer::F32(PyBuffer::get(array)?),
            Dtype::F64 => Buffer::F64(PyBuffer::get(array)?),
        };
        let contiguous = match &buffer {
            Buffer::F32(b) => b.is_c_contiguous(),
            Buffer::F64(b) => b.is_c_contiguous(),
        };
        if !contiguous {
            return Err(PyValueError::new_err("tensors must be C contiguous"));
        }
        Ok(buffer)
    }

    /// Borrows the array without copying it.
    ///
    /// SAFETY: the caller keeps `self` alive for as long as the view is used.
    /// `self` holds a buffer export, so the array can't be freed or resized.
    /// The trainer's contract is that nothing writes to a returned array until
    /// its function has been called again.
    unsafe fn view(&self) -> TensorView<'_> {
        unsafe {
            match self {
                Buffer::F32(b) => TensorView::F32(std::slice::from_raw_parts(
                    b.buf_ptr() as *const f32,
                    b.item_count(),
                )),
                Buffer::F64(b) => TensorView::F64(std::slice::from_raw_parts(
                    b.buf_ptr() as *const f64,
                    b.item_count(),
                )),
            }
        }
    }
}

struct AssertSend<T>(T);

// SAFETY: used only to carry the encoder closure and borrowed slices across
// `detach`, which runs them on this same thread and returns before the
// borrows end.
unsafe impl<T> Send for AssertSend<T> {}

impl<T> AssertSend<T> {
    fn into_inner(self) -> T {
        self.0
    }
}

struct PyTrainer {
    callable: Py<PyAny>,
    layout: Layout,
}

impl PyTrainer {
    fn call(
        &mut self,
        py: Python<'_>,
        ctx: &TrainContext,
        submit: &mut Submit<'_>,
    ) -> PyResult<()> {
        let path = ctx.base_model.to_string_lossy().into_owned();
        let output = self.callable.call1(py, (ctx.round, path))?;
        let (arrays, weight): (Vec<Bound<'_, PyAny>>, u64) = output.extract(py)?;
        if arrays.len() != self.layout.tensors.len() {
            return Err(PyValueError::new_err(
                "the trainer returned the wrong tensors",
            ));
        }
        let buffers = arrays
            .iter()
            .zip(&self.layout.tensors)
            .map(|(array, spec)| Buffer::get(array, spec.dtype))
            .collect::<PyResult<Vec<_>>>()?;
        // SAFETY: `buffers` outlives `views`, which is dropped below.
        let views: Vec<TensorView<'_>> = buffers.iter().map(|b| unsafe { b.view() }).collect();
        let job = AssertSend((views, submit));
        let result = py.detach(move || {
            let (views, submit) = job.into_inner();
            submit(&views, weight).map(|_| ())
        });
        result.map_err(fail)
    }
}

impl Trainer for PyTrainer {
    fn train(&mut self, ctx: &TrainContext, submit: &mut Submit<'_>) -> Result<(), String> {
        Python::attach(|py| self.call(py, ctx, submit)).map_err(|e| e.to_string())
    }
}

fn runtime() -> PyResult<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(fail)
}

fn summary_dict<'py>(
    py: Python<'py>,
    result: Result<SiteSummary, NodeError>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    match result {
        Ok(s) => {
            dict.set_item("ok", true)?;
            dict.set_item("final_model", s.final_model.to_string_lossy().into_owned())?;
            dict.set_item("final_digest", s.final_digest.hex())?;
            dict.set_item("rounds_contributed", s.rounds_contributed)?;
            dict.set_item("rounds_refused", s.rounds_refused)?;
        }
        Err(e) => {
            dict.set_item("ok", false)?;
            dict.set_item("error", e.to_string())?;
        }
    }
    Ok(dict)
}

/// A whole federation on one machine, for notebooks and tests.
#[pyclass]
struct Federation {
    inner: LocalFederation,
}

#[pymethods]
impl Federation {
    /// `options` is a JSON object. See `Options::from_json` for its fields.
    #[new]
    fn new(
        root: PathBuf,
        layout: Vec<(String, String, Vec<u64>)>,
        initial_model: PathBuf,
        options: &str,
    ) -> PyResult<Federation> {
        let options =
            Options::from_json(options).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let inner = LocalFederation::create(&root, parse_layout(layout)?, &initial_model, &options)
            .map_err(fail)?;
        Ok(Federation { inner })
    }

    #[getter]
    fn plan_id(&self) -> String {
        self.inner.plan_id.hex()
    }

    #[getter]
    fn site_dids(&self) -> Vec<String> {
        self.inner.plan.sites.clone()
    }

    #[getter]
    fn helper_dids(&self) -> Vec<String> {
        self.inner
            .plan
            .helpers
            .iter()
            .map(|h| h.did.clone())
            .collect()
    }

    #[getter]
    fn aggregator_did(&self) -> String {
        self.inner.plan.aggregator.did.clone()
    }

    #[getter]
    fn log_path(&self) -> String {
        self.inner
            .aggregator_dir()
            .join("log.jsonl")
            .to_string_lossy()
            .into_owned()
    }

    /// Writes each party's files, so the parties can run as separate processes.
    fn save(&self) -> PyResult<()> {
        self.inner.save().map_err(fail)
    }

    /// Runs the aggregator and helpers here and one trainer per site. Each
    /// trainer is called as `trainer(round, base_model_path)` and returns
    /// Has the data access committee take back a site's access grant.
    fn revoke_access(&self, site: usize) -> PyResult<()> {
        self.inner.revoke_access(site).map_err(fail)
    }

    /// `(arrays_in_layout_order, weight)`.
    #[pyo3(signature = (trainers, keep_spool = false))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        trainers: Vec<Py<PyAny>>,
        keep_spool: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        if trainers.len() != self.inner.sites.len() {
            return Err(PyValueError::new_err("need one trainer per site"));
        }
        let fed = &self.inner;
        let layout = fed.plan.layout.clone();
        let (site_results, aggregator) = py.detach(|| -> PyResult<_> {
            let runtime = runtime()?;
            Ok(runtime.block_on(async {
                let (tx, rx) = tokio::sync::oneshot::channel();
                let mut config = fed.aggregator_config("127.0.0.1:0");
                config.keep_spool = keep_spool;
                config.on_ready = Some(Box::new(move |addr| {
                    let _ = tx.send(addr);
                }));
                let aggregator = tokio::spawn(run_aggregator(config));
                let Ok(addr) = rx.await else {
                    let error = aggregator.await.map_or_else(
                        |e| e.to_string(),
                        |r| r.err().map_or_else(String::new, |e| e.to_string()),
                    );
                    return (Vec::new(), Some(error));
                };
                for h in 0..fed.helpers.len() {
                    tokio::spawn(run_helper(fed.helper_config(h, &addr)));
                }
                let sites: Vec<_> = trainers
                    .into_iter()
                    .enumerate()
                    .map(|(i, callable)| {
                        let trainer = PyTrainer {
                            callable,
                            layout: layout.clone(),
                        };
                        tokio::spawn(run_site(fed.site_config(i, &addr), Box::new(trainer)))
                    })
                    .collect();
                let mut results = Vec::with_capacity(sites.len());
                for site in sites {
                    results.push(match site.await {
                        Ok(result) => result,
                        Err(e) => Err(NodeError::Join(e.to_string())),
                    });
                }
                let finished = tokio::time::timeout(Duration::from_secs(60), aggregator).await;
                let error = match finished {
                    Ok(Ok(Ok(_))) => None,
                    Ok(Ok(Err(e))) => Some(e.to_string()),
                    Ok(Err(e)) => Some(e.to_string()),
                    Err(_) => Some("the aggregator did not stop".to_owned()),
                };
                (results, error)
            }))
        })?;
        let out = PyDict::new(py);
        let sites = site_results
            .into_iter()
            .map(|r| summary_dict(py, r))
            .collect::<PyResult<Vec<_>>>()?;
        out.set_item("sites", sites)?;
        out.set_item("aggregator_error", aggregator)?;
        Ok(out)
    }
}

/// Runs one site until the run ends. For deployments, where every party is
/// its own process.
#[pyfunction]
#[pyo3(signature = (
    key, plan, grant, connect, dir, trainer, layout, revocations = None, give_up_secs = 600,
    standing = None, status_dir = None, registry_rpc = None
))]
#[allow(clippy::too_many_arguments)]
fn run_site_process<'py>(
    py: Python<'py>,
    key: PathBuf,
    plan: PathBuf,
    grant: PathBuf,
    connect: String,
    dir: PathBuf,
    trainer: Py<PyAny>,
    layout: Vec<(String, String, Vec<u64>)>,
    revocations: Option<PathBuf>,
    give_up_secs: u64,
    standing: Option<PathBuf>,
    status_dir: Option<PathBuf>,
    registry_rpc: Option<String>,
) -> PyResult<Bound<'py, PyDict>> {
    let layout = parse_layout(layout)?;
    let identity = Arc::new(Identity::load(&key).map_err(fail)?);
    let plan_line = std::fs::read(&plan).map_err(fail)?;
    let grant = GrantFile::load(&grant).map_err(fail)?;
    let standing = match standing {
        Some(path) => StandingFile::load(&path, status_dir).map_err(fail)?,
        None => Default::default(),
    };
    let config = SiteConfig {
        identity,
        plan_line,
        addr: connect,
        dir,
        invocation: grant.invocation,
        proofs: grant.proofs,
        revocations: revocations.map(|path| Arc::new(RevocationFile(path)) as _),
        give_up_after: Duration::from_secs(give_up_secs),
        standing,
        registry: registry_rpc.map(|url| Arc::new(RpcRegistry { url }) as _),
    };
    let trainer = PyTrainer {
        callable: trainer,
        layout,
    };
    let result = py.detach(|| -> PyResult<_> {
        Ok(runtime()?.block_on(run_site(config, Box::new(trainer))))
    })?;
    summary_dict(py, result)
}

/// Replays a log from nothing. Returns the report as JSON, or raises
/// `VerifyFailed` naming the entry that broke a rule.
#[pyfunction]
#[pyo3(signature = (log, model = None))]
fn verify(py: Python<'_>, log: PathBuf, model: Option<PathBuf>) -> PyResult<(String, bool)> {
    let checked: Result<_, VerifyError> = py.detach(|| verify_log(&log, model.as_deref()));
    let report = checked.map_err(|e| VerifyFailed::new_err(e.to_string()))?;
    let complete = report.complete();
    Ok((serde_json::to_string(&report).map_err(fail)?, complete))
}

/// Writes the first model of a run.
#[pyfunction(name = "write_model")]
fn write_model_py(
    path: &str,
    layout: Vec<(String, String, Vec<u64>)>,
    arrays: Vec<Bound<'_, PyAny>>,
) -> PyResult<String> {
    let layout = parse_layout(layout)?;
    let buffers = arrays
        .iter()
        .zip(&layout.tensors)
        .map(|(array, spec)| Buffer::get(array, spec.dtype))
        .collect::<PyResult<Vec<_>>>()?;
    // SAFETY: `buffers` outlives `views`, and the GIL stays held.
    let views: Vec<TensorView<'_>> = buffers.iter().map(|b| unsafe { b.view() }).collect();
    write_model(Path::new(path), &layout, &views).map_err(fail)?;
    fedtally_core::vector::Digest::file(Path::new(path))
        .map(|d| d.hex())
        .map_err(fail)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Federation>()?;
    m.add_function(wrap_pyfunction!(run_site_process, m)?)?;
    m.add_function(wrap_pyfunction!(verify, m)?)?;
    m.add_function(wrap_pyfunction!(write_model_py, m)?)?;
    m.add("FedtallyError", m.py().get_type::<FedtallyError>())?;
    m.add("VerifyFailed", m.py().get_type::<VerifyFailed>())?;
    Ok(())
}
