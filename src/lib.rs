use thiserror::Error;
use tokio::sync;
use tokio::task::JoinHandle;

use crate::launched::{CacheMessage, LaunchedCache};

mod launched;

type Idx = usize;

#[derive(Error, Debug)]
pub enum MappingError {
    #[error("io error")]
    Io(#[from] std::io::Error),
}

#[derive(Clone, Debug)]
pub struct Entry<T> {
    pub value: T,
    pub embedding: Vec<f32>,
}

#[derive(Debug)]
pub struct SimilarityEntry<T> {
    pub entry: Entry<T>,
    pub similarity: f32,
}

pub struct Cache<T> {
    tx: sync::mpsc::Sender<CacheMessage<T>>,
    // TODO: in the future use handle
    _handle: JoinHandle<()>,
}

impl<T> From<Vec<Entry<T>>> for Cache<T>
where
    T: Send + Sync + Clone + 'static,
{
    fn from(entries: Vec<Entry<T>>) -> Self {
        let (tx, rx) = sync::mpsc::channel(100);

        let handle = tokio::spawn({
            let tx = tx.clone();
            async move {
                let cache = LaunchedCache::new(tx, rx, entries);
                cache.run().await;
            }
        });

        Self {
            tx,
            _handle: handle,
        }
    }
}

// TODO: why do we need Sync bound
impl<T> Cache<T>
where
    T: Send + Sync + Clone + 'static,
{
    /// # Errors
    /// - returns [`MappingError`]
    pub fn new() -> Result<Self, MappingError> {
        let (tx, rx) = sync::mpsc::channel(100);

        let handle = tokio::spawn({
            let tx = tx.clone();
            async move {
                let cache = LaunchedCache::new(tx, rx, vec![]);
                cache.run().await;
            }
        });

        Ok(Self {
            tx,
            _handle: handle,
        })
    }

    /// # Errors
    /// - error if the message could not be sent
    pub async fn rebuild(&self) -> anyhow::Result<bool> {
        let (tx, rx) = sync::oneshot::channel();

        self.tx
            .send(CacheMessage::Rebuild(tx))
            .await
            .map_err(|_| anyhow::anyhow!("could not send message"))?;

        let res = rx
            .await
            .map_err(|_| anyhow::anyhow!("could not receive message"))?;

        Ok(res)
    }

    pub async fn get_by_id(&self, id: Idx) -> Option<Entry<T>> {
        let (tx, rx) = sync::oneshot::channel();
        if self.tx.send(CacheMessage::GetById(id, tx)).await.is_err() {
            return None;
        }
        rx.await.ok().flatten()
    }

    pub async fn get_closest(&self, n: usize, embedding: Vec<f32>) -> Vec<SimilarityEntry<T>> {
        let (tx, rx) = sync::oneshot::channel();
        if self
            .tx
            .send(CacheMessage::Closest { n, embedding, tx })
            .await
            .is_err()
        {
            return vec![];
        }
        rx.await.ok().unwrap_or_default()
    }

    /// # Errors
    /// - error if the message could not be sent
    pub async fn add(&self, key: T, embedding: Vec<f32>) -> anyhow::Result<()> {
        self.tx
            .send(CacheMessage::Add {
                key,
                embedding,
            })
            .await
            .map_err(|_| anyhow::anyhow!("failed to send message"))?;
        Ok(())
    }
}
