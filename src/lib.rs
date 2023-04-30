use std::sync::Arc;

use anyhow::bail;
use async_trait::async_trait;
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

#[async_trait]
trait Embeddable {
    type Error;
    async fn embed(&self) -> Result<Vec<f32>, Self::Error>;
}

#[derive(Clone, Debug)]
pub struct Key<K> {
    pub value: K,
    pub embedding: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct Entry<K, V> {
    pub key: Key<K>,
    pub value: V,
}

#[derive(Debug)]
pub struct SimilarityEntry<K, V> {
    pub entry: Arc<Entry<K, V>>,
    pub similarity: f32,
}

pub struct Cache<K, V> {
    tx: sync::mpsc::Sender<CacheMessage<K, V>>,
    // TODO: in the future use handle
    _handle: JoinHandle<()>,
}

impl<K, V> From<Vec<Entry<K, V>>> for Cache<K, V>
where
    K: Send + Sync + 'static,
    V: Send + Sync + 'static,
{
    fn from(entries: Vec<Entry<K, V>>) -> Self {
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
impl<K: Send + Sync + 'static, V: Send + Sync + 'static> Cache<K, V> {
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

        let res = rx.await.map_err(|_| anyhow::anyhow!("could not receive message"))?;

        Ok(res)
    }

    pub async fn get_by_id(&self, id: Idx) -> Option<Arc<Entry<K, V>>> {
        let (tx, rx) = sync::oneshot::channel();
        if self.tx.send(CacheMessage::GetById(id, tx)).await.is_err() {
            return None;
        }
        rx.await.ok().flatten()
    }

    pub async fn get_closest(&self, n: usize, embedding: Vec<f32>) -> Vec<SimilarityEntry<K, V>> {
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
    pub async fn add(&self, key: K, embedding: Vec<f32>, value: V) -> anyhow::Result<()> {
        self.tx
            .send(CacheMessage::Add {
                key,
                embedding,
                value,
            })
            .await
            .map_err(|_| anyhow::anyhow!("failed to send message"))?;
        Ok(())
    }
}
