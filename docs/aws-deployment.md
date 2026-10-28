# Manual AWS deployment — not executed

Target: one Linux EC2 host with persistent storage running the supplied Compose stack.
This is a learning deployment, not a highly available production architecture.

1. Provision an EC2 instance and encrypted EBS storage with enough memory for Postgres, Redis,
   the API and optionally Kafka. Choose size after measuring your workload; no cost estimate is claimed.
2. Install Docker Engine and the Compose plugin. Copy the repository without `.env`, local runtime
   state or `.git` history. Create `.env` on the host with restricted permissions.
3. Set a strong `POSTGRES_PASSWORD`, your OpenAI key, `COOKIE_SECURE=true` and a suitable rate limit.
   Initially permit registration only while creating your account; then set `REGISTRATION_ENABLED=false`
   and restart the API. Do not expose an open registration page to the internet.
4. Run the Compose stack, including Kafka only if required. Check migration success, then test uploads,
   account isolation, a supported answer and a refusal. `/api/v1/health` only proves process liveness.
5. Put an HTTPS reverse proxy on the host in front of `127.0.0.1:8080`. Permit only HTTPS and restricted
   administration access in the security group. Database, Redis and Kafka ports stay private.
6. Configure consistent backups of PostgreSQL, the SQLite application database and uploaded files.
   Stop writers or use SQLite's backup API; copying a live database file alone is not a safe backup.
7. Verify restoration on a separate instance before making durability claims. Configure logs and disk
   alerts. Stop resources when finished to avoid ongoing charges.

The provided Publish container workflow pushes an image to GHCR when manually run on GitHub.
It does not deploy to AWS. A production evolution would move account/job state into PostgreSQL,
uploads into S3, add stronger authentication, distributed locks and managed infrastructure.

References: [Compose startup dependencies](https://docs.docker.com/compose/how-tos/startup-order/),
[Kafka Docker setup](https://kafka.apache.org/41/getting-started/docker/).
