use anyhow::Context;
use embed_db::{Cache, Entry, Key};

#[tokio::test]
async fn test_new() -> anyhow::Result<()> {
    let cache = Cache::new()?;

    cache.add("hello", vec![1.0, 0.0, 0.0], "world").await?;

    let first = cache.get_by_id(0).await.context("failed to get by id")?;

    assert_eq!(first.key.value, "hello");

    let res = cache.get_closest(1, vec![1.0, 0.0, 0.0]).await;

    assert_eq!(res.len(), 1);
    assert_eq!(res[0].entry.key.value, "hello");

    cache.rebuild().await?;

    let res = cache.get_closest(1, vec![1.0, 0.0, 0.0]).await;

    assert_eq!(res.len(), 1);
    assert_eq!(res[0].entry.key.value, "hello");

    Ok(())
}

#[tokio::test]
async fn test_from_existing() {
    let entry = Entry {
        key: Key {
            value: "hello",
            embedding: vec![0.7, 0.0, 0.7],
        },
        value: "world",
    };

    let cache = Cache::from(vec![entry]);

    let first = cache
        .get_by_id(0)
        .await
        .context("failed to get by id")
        .unwrap();

    assert_eq!(first.key.value, "hello");

    let res = cache.get_closest(1, vec![0.5777, 0.5777, 0.5777]).await;

    assert_eq!(res.len(), 1);

    let first = res.first().unwrap();

    assert_eq!(first.entry.key.value, "hello");
    println!("similarity: {}", first.similarity);
}
