use float_ord::FloatOrd;
use rannoy::Rannoy;
use tokio::sync;

use crate::{Entry, Idx, Key, SimilarityEntry};

pub struct LaunchedCache<K, V> {
    tx: sync::mpsc::Sender<CacheMessage<K, V>>,
    rx: sync::mpsc::Receiver<CacheMessage<K, V>>,
    is_rebuilding: bool,
    indexed_entries: Vec<Entry<K, V>>,
    indexed: Rannoy,
    unindexed: Vec<Entry<K, V>>,
}

pub enum CacheMessage<K, V> {
    /// get an entry by ID
    GetById(Idx, sync::oneshot::Sender<Option<Entry<K, V>>>),
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
    Rebuild(sync::oneshot::Sender<bool>),
    UpdateDataPostRebuild {
        indexed_entries: Vec<Entry<K, V>>,
        indexed: Rannoy,
        tx: sync::oneshot::Sender<bool>,
    },
}

pub fn build<'a, K: 'a, V: 'a>(entries: impl IntoIterator<Item = &'a Entry<K, V>>) -> Rannoy {
    let entries = entries.into_iter().collect::<Vec<_>>();

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

impl<K, V> LaunchedCache<K, V>
where
    K: Send + Sync + Clone + 'static,
    V: Send + Sync + Clone + 'static,
{
    pub fn new(
        tx: sync::mpsc::Sender<CacheMessage<K, V>>,
        rx: sync::mpsc::Receiver<CacheMessage<K, V>>,
        entries: Vec<Entry<K, V>>,
    ) -> Self {
        let annoy = build(&entries);

        Self {
            tx,
            rx,
            is_rebuilding: false,
            indexed_entries: entries,
            indexed: annoy,
            unindexed: vec![],
        }
    }

    pub async fn run(mut self) {
        while let Some(msg) = self.rx.recv().await {
            self.process_message(msg);
        }
    }
}

impl<K, V> LaunchedCache<K, V>
where
    K: Send + Sync + Clone + 'static,
    V: Send + Sync + Clone + 'static,
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

                self.unindexed.push(entry);
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
