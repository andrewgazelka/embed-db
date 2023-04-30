use anyhow::Context;
use embed_db::{Cache, Entry, Key};

#[tokio::test]
async fn test_new() -> anyhow::Result<()> {
    let cache = Cache::new()?;

    cache.add("hello", vec![1.0, 2.0, 3.0], "world").await?;

    let first = cache.get_by_id(0).await.context("failed to get by id")?;

    assert_eq!(first.key.value, "hello");

    let res = cache.get_closest(1, vec![1.0, 2.0, 3.0]).await;

    assert_eq!(res.len(), 1);
    assert_eq!(res[0].key.value, "hello");

    cache.rebuild().await?;

    let res = cache.get_closest(1, vec![1.0, 2.0, 3.0]).await;

    assert_eq!(res.len(), 1);
    assert_eq!(res[0].key.value, "hello");

    Ok(())
}

#[tokio::test]
async fn test_from_existing() {
    let entry = Entry {
        key: Key {
            value: "hello",
            embedding: vec![1.0, 2.0, 3.0],
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

    let res = cache.get_closest(1, vec![1.0, 2.0, 3.0]).await;

    assert_eq!(res.len(), 1);
    assert_eq!(res[0].key.value, "hello");
}
