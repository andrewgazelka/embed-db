use std::sync::Arc;

use float_ord::FloatOrd;
use rannoy::Rannoy;
use tokio::sync;

use crate::{Entry, Idx, Key, SimilarityEntry};

pub struct LaunchedCache<K, V> {
    rx: sync::mpsc::Receiver<CacheMessage<K, V>>,
    data: Data<K, V>,
}

pub enum CacheMessage<K, V> {
    /// get an entry by ID
    GetById(Idx, sync::oneshot::Sender<Option<Arc<Entry<K, V>>>>),
    /// get closest entry
    Closest {
        n: usize,
        embedding: Vec<f32>,
        tx: sync::oneshot::Sender<Vec<SimilarityEntry<K, V>>>,
    },
    Add {
        key: K,
        embedding: Vec<f32>,
        value: V,
    },
    GetDataPreRebuild {
        #[allow(clippy::type_complexity)]
        tx: sync::oneshot::Sender<Option<Vec<Arc<Entry<K, V>>>>>,
    },
    UpdateDataPostRebuild {
        indexed_entries: Vec<Arc<Entry<K, V>>>,
        indexed: Rannoy,
    },
}

pub struct Data<K, V> {
    is_rebuilding: bool,
    indexed_entries: Vec<Arc<Entry<K, V>>>,
    indexed: Rannoy,
    unindexed: Vec<Arc<Entry<K, V>>>,
}

pub fn build<'a, K: 'a, V: 'a>(entries: impl IntoIterator<Item = &'a Entry<K, V>>) -> Rannoy {
    let entries = entries.into_iter().collect::<Vec<_>>();

    // let first_entry: &Entry<K,V> = entries.get(0)?;
    // let dim = first_entry.key.embedding.len();

    let size = entries.len();
    let annoy = Rannoy::new(size);

    for (i, entry) in entries.into_iter().enumerate() {
        let i = i32::try_from(i).unwrap();
        let embedding = entry.key.embedding.clone();
        annoy.add_item(i, &embedding);
    }

    // 30 trees
    annoy.build(30);
    annoy
}

impl<K: Send + 'static, V: Send + 'static> LaunchedCache<K, V> {
    pub fn new(rx: sync::mpsc::Receiver<CacheMessage<K, V>>, entries: Vec<Entry<K, V>>) -> Self {
        let annoy = build(&entries);

        Self {
            rx,
            data: Data {
                is_rebuilding: false,
                indexed_entries: entries.into_iter().map(Arc::new).collect(),
                indexed: annoy,
                unindexed: vec![],
            },
        }
    }

    pub async fn run(mut self) {
        while let Some(msg) = self.rx.recv().await {
            self.data.process_message(msg);
        }
    }
}

impl<K, V> Data<K, V> {
    fn process_message(&mut self, message: CacheMessage<K, V>) {
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
                        let entry_embedding = &entry.key.embedding;
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
            CacheMessage::Add {
                key,
                value,
                embedding,
            } => {
                let entry = Entry {
                    key: Key {
                        value: key,
                        embedding,
                    },
                    value,
                };

                self.unindexed.push(entry.into());
            }
            CacheMessage::GetDataPreRebuild { tx } => {
                if self.is_rebuilding {
                    let _ = tx.send(None);
                    return;
                }
                self.is_rebuilding = true;
                let mut indexed = self.indexed_entries.clone();
                indexed.extend(self.unindexed.clone());
                let _ = tx.send(Some(indexed));
            }
            CacheMessage::UpdateDataPostRebuild {
                indexed,
                indexed_entries,
            } => {
                let prev_len = self.indexed_entries.len();
                let new_len = indexed_entries.len();
                let diff_len = new_len - prev_len;

                self.indexed = indexed;
                self.indexed_entries = indexed_entries;
                self.unindexed.drain(0..diff_len);

                self.is_rebuilding = false;
            }
        }
    }
}
