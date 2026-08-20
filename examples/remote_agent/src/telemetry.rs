use aquaregia::AgentStreamEvent;

use crate::api::agent_event_name;
use crate::store::{Store, StoreResult};

pub async fn record(
    store: &Store,
    run_id: &str,
    seq: i64,
    event: &AgentStreamEvent,
) -> StoreResult<()> {
    let payload = serde_json::to_value(event)
        .map_err(|err| crate::store::StoreError::new(err.to_string()))?;
    store
        .insert_event(run_id, seq, agent_event_name(event), &payload)
        .await
}
