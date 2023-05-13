use float_ord::FloatOrd;
use rannoy::Rannoy;
use std::num::NonZeroUsize;
use tokio::sync;

use crate::{Entry, Idx, SimilarityEntry};

pub struct CacheActor<T> {
    tx: sync::mpsc::Sender<CacheMessage<T>>,
    rx: sync::mpsc::Receiver<CacheMessage<T>>,
    is_rebuilding: bool,
    indexed_entries: Vec<Entry<T>>,
    indexed: Rannoy,
    unindexed: Vec<Entry<T>>,
    rebuild_threshold: Option<NonZeroUsize>,
}

pub enum CacheMessage<T> {
    /// get an entry by ID
    GetById(Idx, sync::oneshot::Sender<Option<Entry<T>>>),
    /// get closest entry
    Closest {
        n: usize,
        embedding: Vec<f32>,
        tx: sync::oneshot::Sender<Vec<SimilarityEntry<T>>>,
    },
    Add {
        key: T,
        embedding: Vec<f32>,
    },
    Rebuild(sync::oneshot::Sender<bool>),
    UpdateDataPostRebuild {
        indexed_entries: Vec<Entry<T>>,
        indexed: Rannoy,
        tx: sync::oneshot::Sender<bool>,
    },
}

pub fn build<'a, T: 'a>(entries: impl IntoIterator<Item = &'a Entry<T>>) -> Rannoy {
    let entries = entries.into_iter().collect::<Vec<_>>();

    let size = entries.len();
    let annoy = Rannoy::new(size);

    for (i, entry) in entries.into_iter().enumerate() {
        let i = i32::try_from(i).unwrap();
        let embedding = entry.embedding.clone();
        annoy.add_item(i, &embedding);
    }

    // 30 trees
    annoy.build(30);
    annoy
}

impl<T> CacheActor<T>
where
    T: Send + Sync + Clone + 'static,
{
    pub fn new(
        tx: sync::mpsc::Sender<CacheMessage<T>>,
        rx: sync::mpsc::Receiver<CacheMessage<T>>,
        entries: Vec<Entry<T>>,
        rebuild_threshold: Option<NonZeroUsize>,
    ) -> Self {
        let annoy = build(&entries);

        Self {
            tx,
            rx,
            is_rebuilding: false,
            indexed_entries: entries,
            indexed: annoy,
            unindexed: vec![],
            rebuild_threshold,
        }
    }

    pub async fn run(mut self) {
        while let Some(msg) = self.rx.recv().await {
            self.process_message(msg);
        }
    }
}

impl<T> CacheActor<T>
where
    T: Send + Sync + Clone + 'static,
{
    fn rebuild(&mut self, tx: sync::oneshot::Sender<bool>) {
        if self.is_rebuilding {
            let _ = tx.send(false);
            return;
        }
        self.is_rebuilding = true;

        let mut indexed = self.indexed_entries.clone();
        indexed.extend(self.unindexed.clone());

        let self_tx = self.tx.clone();

        tokio::spawn(async move {
            let (annoy, data) = tokio::task::spawn_blocking({
                move || {
                    let build = build(&indexed);
                    (build, indexed)
                }
            })
            .await
            .unwrap_or_else(|_| panic!("failed to spawn blocking task"));

            self_tx
                .send(CacheMessage::UpdateDataPostRebuild {
                    indexed_entries: data,
                    indexed: annoy,
                    tx,
                })
                .await
                .unwrap_or_else(|_| panic!("failed to send message"));
        });
    }
    fn process_message(&mut self, message: CacheMessage<T>) {
        match message {
            CacheMessage::GetById(id, tx) => {
                let entry = self.indexed_entries.get(id).cloned();
                let entry = if entry.is_none() {
                    let indexed_id = id - self.indexed_entries.len();
                    self.unindexed.get(indexed_id).cloned()
                } else {
                    entry
                };

                let _ = tx.send(entry);
            }
            CacheMessage::Closest { n, embedding, tx } => {
                let embedding_arr = ndarray::arr1(&embedding);

                let idxs = match self.indexed_entries.is_empty() {
                    true => vec![],
                    false => {
                        self.indexed
                            .get_nns_by_vector(embedding, i32::try_from(n).unwrap(), 30)
                            .0
                    }
                };

                let unindexed_len = self.unindexed.len();
                let mut res = Vec::with_capacity(n + unindexed_len);

                for idx in idxs {
                    let idx = usize::try_from(idx).unwrap();
                    res.push(self.indexed_entries.get(idx).unwrap().clone());
                }

                res.extend(self.unindexed.clone());

                // sort by distance to embedding

                let mut res = res
                    .into_iter()
                    .map(|entry| {
                        let entry_embedding = &entry.embedding;
                        let entry_embedding = ndarray::arr1(entry_embedding);

                        let dot = embedding_arr.dot(&entry_embedding);

                        SimilarityEntry {
                            entry,
                            similarity: dot,
                        }
                    })
                    .collect::<Vec<_>>();

                res.sort_unstable_by_key(|entry| FloatOrd(-entry.similarity));

                res.truncate(n);

                let _ = tx.send(res);
            }
            CacheMessage::Add { key, embedding } => {
                let entry = Entry {
                    value: key,
                    embedding,
                };

                self.unindexed.push(entry);

                if let Some(threshold) = self.rebuild_threshold {
                    if self.unindexed.len() >= threshold.get() {
                        // unused channel. TODO: is this a good practice?
                        let (tx, _) = sync::oneshot::channel();

                        self.rebuild(tx);
                    }
                }
            }
            CacheMessage::Rebuild(tx) => {
                self.rebuild(tx);
            }
            CacheMessage::UpdateDataPostRebuild {
                indexed,
                indexed_entries,
                tx,
            } => {
                let prev_len = self.indexed_entries.len();
                let new_len = indexed_entries.len();
                let diff_len = new_len - prev_len;

                self.indexed = indexed;
                self.indexed_entries = indexed_entries;
                self.unindexed.drain(0..diff_len);

                self.is_rebuilding = false;

                let _ = tx.send(true);
            }
        }
    }
}
