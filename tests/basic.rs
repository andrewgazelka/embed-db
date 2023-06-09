use std::num::NonZeroUsize;

use anyhow::Context;

use embed_db::{Cache, Entry};

#[tokio::test]
async fn test_new() -> anyhow::Result<()> {
    let cache = Cache::new(None)?;

    cache.add("hello", vec![1.0, 0.0, 0.0]).await?;

    let first = cache.get_by_id(0).await.context("failed to get by id")?;

    assert_eq!(first.value, "hello");

    let res = cache.get_closest(1, vec![1.0, 0.0, 0.0]).await;

    assert_eq!(res.len(), 1);
    assert_eq!(res[0].entry.value, "hello");

    cache.rebuild().await?;

    let res = cache.get_closest(1, vec![1.0, 0.0, 0.0]).await;

    assert_eq!(res.len(), 1);
    assert_eq!(res[0].entry.value, "hello");

    Ok(())
}

#[tokio::test]
async fn test_from_existing() {
    let entry = Entry {
        value: "hello",
        embedding: vec![0.7, 0.0, 0.7],
    };

    let cache = Cache::new_from_entries(vec![entry], NonZeroUsize::new(123));

    let first = cache
        .get_by_id(0)
        .await
        .context("failed to get by id")
        .unwrap();

    assert_eq!(first.value, "hello");

    let res = cache.get_closest(1, vec![0.5777, 0.5777, 0.5777]).await;

    assert_eq!(res.len(), 1);

    let first = res.first().unwrap();

    assert_eq!(first.entry.value, "hello");
    println!("similarity: {}", first.similarity);
}
