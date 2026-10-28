"""Durable outbox publisher and Kafka ingestion consumer.

Run with: python -m stacklens.workspace.worker
Only opaque job IDs travel through Kafka. Ownership comes from the database.
"""

from __future__ import annotations

import json
import logging

from confluent_kafka import Consumer, Producer

from stacklens.config import get_settings
from stacklens.workspace import store
from stacklens.workspace.ingestion import process_job

logger = logging.getLogger(__name__)


def publish_pending(producer) -> int:
    count = 0
    for job in store.pending_jobs():
        errors: list[object] = []
        producer.produce(
            get_settings().kafka_topic,
            key=job["id"],
            value=json.dumps({"job_id": job["id"]}),
            on_delivery=lambda error, _message, errors=errors: (
                errors.append(error) if error else None
            ),
        )
        if producer.flush(10) or errors:
            raise RuntimeError("Kafka delivery was not acknowledged; outbox remains pending.")
        store.mark_published(job["id"])
        count += 1
    return count


def handle_event(payload: bytes) -> bool:
    try:
        event = json.loads(payload)
        job_id = event["job_id"]
        if (
            not isinstance(job_id, str)
            or len(job_id) != 32
            or any(c not in "0123456789abcdef" for c in job_id)
        ):
            raise ValueError("Invalid job ID")
    except (ValueError, KeyError, TypeError):
        logger.warning("Discarded malformed ingestion event.")
        return False
    return process_job(job_id)


def main():
    logging.basicConfig(level=logging.INFO)
    s = get_settings()
    if s.ingestion_mode != "kafka":
        raise SystemExit("Set INGESTION_MODE=kafka for the Kafka worker.")
    producer = Producer(
        {
            "bootstrap.servers": s.kafka_bootstrap_servers,
            "enable.idempotence": True,
            "message.timeout.ms": 10000,
        }
    )
    consumer = Consumer(
        {
            "bootstrap.servers": s.kafka_bootstrap_servers,
            "group.id": s.kafka_group_id,
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
            "max.poll.interval.ms": 1_800_000,
        }
    )
    consumer.subscribe([s.kafka_topic])
    try:
        while True:
            try:
                publish_pending(producer)
            except Exception:
                logger.warning("Kafka publish unavailable; pending jobs retained.")
            message = consumer.poll(1)
            if message is None or message.error():
                continue
            # Unexpected exceptions leave the offset uncommitted for redelivery.
            handle_event(message.value())
            consumer.commit(message=message, asynchronous=False)
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
